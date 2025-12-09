import numpy as np
from scipy.stats import t as t_dist, norm

def robust_se(model, apply, type):

    X = model.X
    THETA = model.theta
    ALPHA = model.alpha
    n, k = X.shape
    model_type = model.model_type

    if model_type == "linear":
        XTX_INV = model.xtx_inv
        DF = model.degrees_freedom
        h = np.sum(X @ XTX_INV * X, axis=1)
        stat_dist = t_dist
        stat_name = 't'
        sr = model.residuals.reshape(-1, 1).flatten()**2

    if model_type == "logit":
        XTX_INV = model.xtWx_inv
        DF = model.degrees_freedom
        h = np.sum(X @ XTX_INV * X, axis=1)
        stat_dist = norm
        stat_name = 'z'
        # Response residuals != deviance residuals
        z = X @ THETA
        mu = 1 / (1 + np.exp(-z))
        response_residuals = model.y - mu
        sr = response_residuals**2

    HC_ = {
        "HC0": lambda sr, n_obs, k_regressors, leverage: sr,
        "HC1": lambda sr, n_obs, k_regressors, leverage: (n_obs / (n_obs - k_regressors)) * sr,
        "HC2": lambda sr, n_obs, k_regressors, leverage: sr / (1 - leverage),
        "HC3": lambda sr, n_obs, k_regressors, leverage: sr / ((1 - leverage) ** 2),
    }

    try:
        omega_diagonal = HC_[type](sr, n, k, h)
        X_omega = X * np.sqrt(omega_diagonal)[:, None]                   # Multiply each X row by X*(diagonal weights)^(0.5)
        robust_cov = XTX_INV @ (X_omega.T @ X_omega) @ XTX_INV             # Sandwich
        robust_se = np.sqrt(np.diag(robust_cov))                              # Diagonal extract the var-cov
        robust_stat = THETA / robust_se

        if model_type == "linear":
            robust_p = 2 * (1 - stat_dist.cdf(abs(robust_stat), DF))
            t_crit = stat_dist.ppf(1 - ALPHA/2, DF)

        if model_type == "logit":
            robust_p = 2 * (1 - stat_dist.cdf(abs(robust_stat)))
            t_crit = stat_dist.ppf(1 - ALPHA/2)


        robust_ci_low = THETA - t_crit * robust_se
        robust_ci_high = THETA + t_crit * robust_se

        if apply:
            model.variance_coefficient = robust_cov
            model.std_error_coefficient = robust_se

            if model_type == "linear":
                model.t_stat_coefficient = robust_stat

            if model_type == "logit":
                model.z_stat_coefficient = robust_stat

            model.p_value_coefficient = robust_p
            model.ci_low = robust_ci_low
            model.ci_high = robust_ci_high

        return {
        "feature":              model.feature_names,
        "robust_se":            robust_se,
        f"robust_{stat_name}":  robust_stat,
        "robust_p":             robust_p,
        f"ci_low_{ALPHA}":      robust_ci_low,
        f"ci_high_{ALPHA}":     robust_ci_high,
    }
    
    except KeyError:
        raise ValueError("Select 'HC0', 'HC1', 'HC2', 'HC3'")