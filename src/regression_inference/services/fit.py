from ..utils.input_validator import validate
import warnings
import numpy as np
from scipy.stats import t as t_dist, norm

def _fit(
        model,
        X:              np.ndarray,
        y:              np.ndarray,
        feature_names:  list[str],
        target_name:    str,
        alpha:          float,
        max_iter:       int = 100,
        tol:            float = 1e-8,
    ):
    X_array, y_array = validate(X, y, alpha, model.model_type)

    # Names override -> Pandas column names -> Generic in place names.
    model.feature_names = (
        ['const', *feature_names] if feature_names is not None
        else X.columns if hasattr(X, 'columns')
        else ['const', *[f"Feature {i}" for i in range(1,X.shape[1])]]
    )
    model.target = (
        target_name if target_name is not None
        else y.name if hasattr(y, 'name')
        else "Dependent"
    )

    model.alpha = alpha
    model.X, model.y = X_array, y_array
    model.degrees_freedom = model.X.shape[0]-model.X.shape[1]

    if model.model_type == "ols":
        
        # Cholesky decomposition fit
        xtx = model.X.T @ model.X
        try:
            L = np.linalg.cholesky(xtx)
            model.theta = np.linalg.solve(L.T, np.linalg.solve(L, model.X.T @ model.y))
            I = np.eye(xtx.shape[0])
            model.xtx_inv = np.linalg.solve(L.T, np.linalg.solve(L, I))

        except np.linalg.LinAlgError:
            raise ValueError(
            "\nMatrix X'X is not positive definite. This typically indicates:\n"
            "- Perfect multicollinearity between features\n"
            "- Insufficient observations (n < k)\n"
            "- Constant or duplicate columns in X"
            )
        
        cond = np.linalg.cond(xtx)
        if cond > 1e10:
            warnings.warn(
                f"\nX'X matrix is ill-conditioned (cond={cond:.2e}).\n"
                f"Results may be unreliable. Consider:\n"
                f"- Removing collinear features\n"
                f"- Scaling features\n",
                UserWarning,
                stacklevel=2
        )
        # Predict
        model.intercept, model.coefficients = model.theta[0], model.theta[1:]
        y_hat = model.X @ model.theta #model.predict(model.X)
        y_bar = np.mean(model.y)
        model.residuals = model.y - y_hat

        # Squared Residuals
        model.rss = model.residuals @ model.residuals
        model.ess = np.sum((y_hat - y_bar)**2)
        model.tss = np.sum((model.y - y_bar)**2)

        # Loss
        model.mse = model.rss / model.degrees_freedom
        model.rmse = np.sqrt(model.mse)

        # Model
        model.f_statistic = (
            (model.ess / model.coefficients.shape[0]) / model.mse
            if model.coefficients.shape[0] > 0 and model.mse > 1e-15
            else np.inf
        )
        model.r_squared = 1 - (model.rss / model.tss)
        model.r_squared_adjusted = 1 - (1 - model.r_squared) * (model.X.shape[0] - 1) / model.degrees_freedom
        model.log_likelihood = -model.X.shape[0]/2 * (np.log(2 * np.pi) + np.log(model.rss / model.X.shape[0]) + 1)
        model.aic = -2 * model.log_likelihood + 2 * model.X.shape[1]
        model.bic = -2 * model.log_likelihood + model.X.shape[1] * np.log(model.X.shape[0])

        # Feature
        model.variance_coefficient = model.mse * model.xtx_inv  # Variance of the coefficients (Covariance matrix)
        model.std_error_coefficient = np.sqrt(np.diag(model.variance_coefficient))
        model.t_stat_coefficient = model.theta / model.std_error_coefficient
        model.p_value_coefficient = 2 * (1 - t_dist.cdf(abs(model.t_stat_coefficient), model.degrees_freedom))
        t_crit = t_dist.ppf(1 - alpha/2, model.degrees_freedom)
        model.ci_low = model.theta - t_crit * model.std_error_coefficient
        model.ci_high = model.theta + t_crit * model.std_error_coefficient

    
    if model.model_type == "mle":

        # Newton-Raphson IRLS for Maximum Likelihood Estimation
        model.theta = np.zeros(model.X.shape[1])

        for _ in range(max_iter):

            z = model.X @ model.theta
            mu = 1 / (1 + np.exp(-z))
            mu = np.clip(mu, 1e-15, 1 - 1e-15)
            gradient = model.X.T @ (mu - model.y)

            # Hessian Fisher Information Matrix
            W = mu * (1 - mu)
            H = model.X.T @ (W[:, np.newaxis] * model.X)

            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "\nHessian matrix is singular. This typically indicates:\n"
                    "- Perfect separation in the data\n"
                    "- Perfect multicollinearity between features\n"
                    "- Insufficient observations\n"
                    "- Constant or duplicate columns in X"
            )
            theta_new = model.theta - H_inv @ gradient
            # convergence
            if np.max(np.abs(theta_new - model.theta)) < tol:
                model.theta = theta_new
                break
            
            model.theta = theta_new
        else:
            warnings.warn(
                f"\nOptimization did not converge after {max_iter} iterations.\n"
                f"Consider:\n"
                f"- Increasing max_iter\n"
                f"- Adjusting tolerance\n"
                f"- Scaling features\n"
                f"- Checking for separation issues\n",
                UserWarning,
                stacklevel=2
        )
        # Predictions on fit
        z = model.X @ model.theta
        mu = 1 / (1 + np.exp(-z))
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        # Recompute Hessian at convergence for inference
        W = mu * (1 - mu)
        H = model.X.T @ (W[:, np.newaxis] * model.X)

        try:
            model.xtWx_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            raise ValueError("Failed to compute covariance matrix at convergence.")

        cond = np.linalg.cond(H)
        if cond > 1e10:
            warnings.warn(
                f"\nHessian matrix is ill-conditioned (cond={cond:.2e}).\n"
                f"Results may be unreliable. Consider:\n"
                f"- Removing collinear features\n"
                f"- Scaling features\n",
                UserWarning,
                stacklevel=2
        )
        # Predict
        model.intercept, model.coefficients = model.theta[0], model.theta[1:]
        y_hat_prob = mu
        y_hat = (y_hat_prob >= 0.5).astype(int)

        # deviance residuals
        model.residuals = np.sign(model.y - y_hat_prob) * np.sqrt(
            -2 * (model.y * np.log(y_hat_prob) + 
                  (1 - model.y) * np.log(1 - y_hat_prob))
        )
        # Log-likelihood
        model.log_likelihood = np.sum(
            model.y * np.log(y_hat_prob) + 
            (1 - model.y) * np.log(1 - y_hat_prob)
        )
        # Deviance/RSS
        model.deviance = -2 * model.log_likelihood
        
        # Null model (intercept only)
        y_bar = np.mean(model.y)
        y_bar = np.clip(y_bar, 1e-15, 1 - 1e-15)
        model.null_log_likelihood = np.sum(
            model.y * np.log(y_bar) + 
            (1 - model.y) * np.log(1 - y_bar)
        )
        model.null_deviance = -2 * model.null_log_likelihood

        # Model metrics
        model.aic = -2 * model.log_likelihood + 2 * model.X.shape[1]
        model.bic = -2 * model.log_likelihood + model.X.shape[1] * np.log(model.X.shape[0])

        # McFadden's Pseudo R-squared
        model.pseudo_r_squared = 1 - (model.log_likelihood / model.null_log_likelihood)

        # Likelihood ratio test statistic analogous to F-statistic
        model.lr_statistic = -2 * (model.null_log_likelihood - model.log_likelihood)

        model.variance_coefficient = model.xtWx_inv  # Covariance matrix
        model.std_error_coefficient = np.sqrt(np.diag(model.variance_coefficient))
        model.z_stat_coefficient = model.theta / model.std_error_coefficient
        model.p_value_coefficient = 2 * (1 - norm.cdf(abs(model.z_stat_coefficient)))

        z_crit = norm.ppf(1 - alpha / 2)
        model.ci_low = model.theta - z_crit * model.std_error_coefficient
        model.ci_high = model.theta + z_crit * model.std_error_coefficient

    return model