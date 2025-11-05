import warnings
import numpy as np
from scipy.stats import t as t_dist
from ..utils.input_validator import validate

def fit(model, X, y, feature_names, target_name, alpha):

    X_array, y_array = validate(X, y, alpha)

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

    # Cholesky decomposition fit
    xtx = model.X.T @ model.X
    try:
        L = np.linalg.cholesky(xtx)
        model.theta = np.linalg.solve(L.T, np.linalg.solve(L, model.X.T @ model.y))
        I = np.eye(xtx.shape[0])
        model.xtx_inv = np.linalg.solve(L.T, np.linalg.solve(L, I))
    except np.linalg.LinAlgError:
        raise ValueError(
        "Matrix X'X is not positive definite. This typically indicates:\n"
        "- Perfect multicollinearity between features\n"
        "- Insufficient observations (n < k)\n"
        "- Constant or duplicate columns in X"
        )
    cond = np.linalg.cond(xtx)
    if cond > 1e10:
        warnings.warn(
            f"X'X matrix is ill-conditioned (cond={cond:.2e}).\n"
            f"Results may be unreliable. Consider:\n"
            f"- Removing collinear features\n"
            f"- Scaling features\n"
            f"- Using regularization\n",
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

    return model