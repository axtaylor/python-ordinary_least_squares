import numpy as np    
from scipy.stats import t as t_dist, norm

def _predict(model, X, alpha, return_table):
        
    if not return_table:
        return (
            (np.asarray(X, dtype=float) @ model.coefficients + model.intercept)
            if model.model_type == "ols" else
            1 / (1 + np.exp(-(np.asarray(X, dtype=float) @ model.coefficients + model.intercept)))
            if model.model_type == "mle" else
            ValueError("Unknown model type")
    )
    prediction_features = {
            name: f'{value_at.item():.2f}'
            for name, value_at in zip(model.feature_names[1:], X[0])
    }
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    prediction = X @ model.theta
    se_prediction = np.sqrt((X @ model.variance_coefficient @ X.T)).item()

    if model.model_type == "ols":

        t_critical = t_dist.ppf(1 - alpha/2, model.degrees_freedom)
        ci_low, ci_high = (prediction - t_critical * se_prediction), (prediction + t_critical * se_prediction)
        t_stat = prediction / se_prediction
        p = 2 * (1 - t_dist.cdf(abs(t_stat), model.degrees_freedom))

        return ({
            "features": [prediction_features],
            "prediction": [np.round(prediction.item(), 4)],
            "std_error": [np.round(se_prediction,4)],
            "t_statistic": [np.round(t_stat.item(),4)],
            "P>|t|": [p.item()],
            f"ci_low_{alpha}": [np.round(ci_low.item(), 4)],
            f"ci_high_{alpha}": [np.round(ci_high.item(), 4)],
        })
    
    if model.model_type == "mle":

        prediction_prob = 1 / (1 + np.exp(-prediction))
        z_critical = norm.ppf(1 - alpha/2)
        ci_low_z, ci_high_z = (prediction - z_critical * se_prediction), (prediction + z_critical * se_prediction)
        ci_low_prob = 1 / (1 + np.exp(-ci_low_z))
        ci_high_prob = 1 / (1 + np.exp(-ci_high_z))
        z_stat = prediction / se_prediction
        p = 2 * (1 - norm.cdf(abs(z_stat)))

        return ({
            "features": [prediction_features],
            "prediction_prob": [np.round(prediction_prob.item(), 4)],
            "prediction_class": [int(prediction_prob.item() >= 0.5)],
            "std_error": [np.round(se_prediction, 4)],
            "z_statistic": [np.round(z_stat.item(), 4)],
            "P>|z|": [p.item()],
            f"ci_low_{alpha}": [np.round(ci_low_prob.item(), 4)],
            f"ci_high_{alpha}": [np.round(ci_high_prob.item(), 4)],
        })
