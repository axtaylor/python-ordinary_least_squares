import numpy as np    
from scipy.stats import t as t_dist

class Predict:

    def predict(model, X, alpha, return_table):

        if return_table == False:
            return (np.asarray(X, dtype=float) @ model.coefficients + model.intercept)

        prediction_features = {j: f'{i.item():.2f}' for j, i in zip(model.feature_names[1:], X[0])}
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        prediction = X @ model.theta
        se_prediction = np.sqrt((X @ model.variance_coefficient @ X.T)).item()
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