import numpy as np
from scipy.stats import t as t_dist

def robust_se(model, type):

    n, k = model.X.shape
    h = np.sum(model.X @ model.xtx_inv * model.X, axis=1)
    sr = model.residuals.reshape(-1, 1).flatten()**2

    HC_ = {
        "HC0": lambda sr, n_obs, k_regressors, leverage: sr,
        "HC1": lambda sr, n_obs, k_regressors, leverage: (n_obs / (n_obs - k_regressors)) * sr,
        "HC2": lambda sr, n_obs, k_regressors, leverage: sr / (1 - leverage),
        "HC3": lambda sr, n_obs, k_regressors, leverage: sr / ((1 - leverage) ** 2),
    }

    try:
        omega_diagonal = HC_[type](sr, n, k, h)
        X_omega = model.X * np.sqrt(omega_diagonal)[:, None]                   # Multiply each X row by X*(diagonal weights)^(0.5)
        robust_cov = model.xtx_inv @ (X_omega.T @ X_omega) @ model.xtx_inv      # Sandwich
        robust_se = np.sqrt(np.diag(robust_cov))                              # Diagonal extract the var-cov
        robust_t_stat = model.theta / robust_se

        return {
        "feature": model.feature_names,
        "robust_se": robust_se,
        "robust_t": robust_t_stat,
        "robust_p": 2 * (1 - t_dist.cdf(abs(robust_t_stat), model.degrees_freedom)),
        #"type": type,
        #"covariance": robust_cov,
    }
    
    except KeyError:
        raise ValueError("Select 'HC0', 'HC1', 'HC2', 'HC3'")