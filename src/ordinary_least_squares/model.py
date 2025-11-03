from dataclasses import dataclass, field
from .services.variance_inflation_factor import variance_inflation_factor
from .services.robust_std_error import robust_se
from .services.hypothesis_testing import hypothesis_testing
from .services.linear_predict import predict
from .services.linear_fit import fit
from .utils.inference_table import inference_table
from .utils.regression_output import summary
import numpy as np

@dataclass
class LinearRegressionOLS:

    alpha: float = None
    feature_names: list = field(default_factory=list)
    target: str = None

    X: np.ndarray = field(default=None, repr=False)
    y: np.ndarray = field(default=None, repr=False)
    
    degrees_freedom: int = None
    xtx_inv: np.ndarray = field(default=None, repr=False)
    theta: np.ndarray = field(default=None)
    coefficients: np.ndarray = field(default=None)
    intercept: float = None
    residuals: np.ndarray = field(default=None, repr=False)

    rss: float = None
    tss: float = None
    ess: float = None
    mse: float = None
    rmse: float = None
    r_squared: float = None
    r_squared_adjusted: float = None
    log_likelihood: float = None
    aic: float = None
    bic: float = None

    # Inference
    variance_coefficient: np.ndarray = field(default=None, repr=False)
    std_error_coefficient: np.ndarray = field(default=None)
    t_stat_coefficient: np.ndarray = field(default=None)
    p_value_coefficient: np.ndarray = field(default=None)
    ci_low: np.ndarray = field(default=None)
    ci_high: np.ndarray = field(default=None)

    def __str__(self):
        self._model_is_fitted()
        return summary(self)

    def fit(self, X, y, feature_names = None, target_name = None, alpha = 0.05):
        return fit(self, X, y, feature_names, target_name, alpha)
    
    def _model_is_fitted(self):
        if self.theta is None:
            raise ValueError("Model is not fitted. Call 'fit' with arguments before using this method.")
    
    def inference_table(self):
        self._model_is_fitted()
        return inference_table(self)

    def predict(self, X, alpha=0.05, return_table=False):
        self._model_is_fitted()
        return predict(self, X, alpha, return_table)

    def hypothesis_testing(self, test, hyp, alpha=0.05):
        self._model_is_fitted()
        return hypothesis_testing(self, test, hyp, alpha)

    def variance_inflation_factor(self):
        self._model_is_fitted()
        return variance_inflation_factor(self)

    def robust_se(self, apply=False, type="HC3"):
        self._model_is_fitted()
        return robust_se(self, apply, type)