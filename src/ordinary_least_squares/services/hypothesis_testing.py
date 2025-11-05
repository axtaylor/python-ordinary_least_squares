import numpy as np
from scipy.stats import t as t_dist

def hypothesis_testing(model, test, hyp, alpha):

    critical = np.round(t_dist.ppf(1 - alpha/2, model.degrees_freedom),2)

    prediction_features = {
        j: f'{i.item():.2f}'
        for j, i in
        zip(model.feature_names[1:], test[0])
    }
    hypothesis_features = (
        {j: f'{i.item():.2f}' for j, i in zip(model.feature_names[1:], hyp[0])}
        if isinstance(hyp, np.ndarray)
        else {f"{model.target}": f"{hyp}"}
    )

    test = np.hstack([np.ones((test.shape[0], 1)), test])

    prediction, hypothesis = test @ model.theta, (
        np.hstack([np.ones((hyp.shape[0], 1)), hyp]) @ model.theta
        if isinstance(hyp, np.ndarray)
        else np.asarray(hyp)
    )

    se = np.sqrt((test @ model.variance_coefficient @ test.T)).item()
    t_stat = (prediction - hypothesis) / se
    p = 2 * (1 - t_dist.cdf(abs(t_stat), model.degrees_freedom))

    result = (
        f"Significance Analysis (p > |t|)\n{critical} > |{t_stat.item():.4f}| == {abs(t_stat.item()) < critical}\n"
        f"\nFail to reject the null hypothesis: {prediction.item():.4f} is not statistically different from {hypothesis.item():.4f} at {alpha*100}% level\n"
        f"\nConclude that outcome of {prediction_features}\ndoes not differ from {hypothesis_features}"
        if abs(t_stat.item()) < critical else
        f"Significance Analysis (p > |t|)\n{critical} > |{t_stat.item():.4f}| == {abs(t_stat.item()) < critical}\n"
        f"\nReject the null hypothesis: {prediction.item():.4f} is statistically different from {hypothesis.item():.4f} at {alpha*100}% level\n"
        f"\nConclude that the outcomes of {prediction_features}\ndiffers significantly from {hypothesis_features}"
    )
    return (
    {
        "summary": result,
        "table": {
            "feature_labels": [prediction_features],
            "hypothesis_labels": [hypothesis_features],
            "prediction": [prediction.item()],
            "hypothesis": [hypothesis.item()],
            "t_statistic": [t_stat.item()],
            "P>|t|": [p.item()],
            },
        }
    )