from dataclasses import dataclass, field
from typing import Union
from abc import ABC, abstractmethod

from .services.variance_inflation_factor import variance_inflation_factor
from .services.robust_std_error import robust_se
from .services.predict import predict
from .services.fit import fit
from .utils.inference_table import inference_table
from .utils.regression_output import summary

import numpy as np

@dataclass
class Model(ABC):

    model_type: str = field(init=False) 
    feature_names: list = field(default_factory=list)
    target: str = None

    X: np.ndarray = field(default=None, repr=False)
    y: np.ndarray = field(default=None, repr=False)

    alpha: float = None
    theta: np.ndarray = field(default=None)
    coefficients: np.ndarray = field(default=None)
    intercept: float = None
    degrees_freedom: int = None
    residuals: np.ndarray = field(default=None, repr=False)
    
    log_likelihood: float = None
    aic: float = None
    bic: float = None

    variance_coefficient: np.ndarray = field(default=None)
    std_error_coefficient: np.ndarray = field(default=None)
    ci_low: np.ndarray = field(default=None)
    ci_high: np.ndarray = field(default=None)

    @property
    def is_fitted(self) -> bool:
        return self.theta is not None
    
    def _model_is_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call 'fit' with arguments before using this method.")
    
    def __str__(self) -> str:
        self._model_is_fitted()
        return summary(self)
    
    @abstractmethod
    def fit(
        self,
        X:              np.ndarray,
        y:              np.ndarray,
        feature_names:  list[str] = None,
        target_name:    str       = None,
        alpha:          float     = 0.05,
    ) -> 'Model':
        
        pass

    def predict(
            self,
            X:              np.ndarray,
            alpha:          float = 0.05,
            return_table:   bool  = False,
    ) -> Union[np.ndarray, dict]:

        self._model_is_fitted()
        return predict(self, X, alpha, return_table)

    def robust_se(self, apply: bool = False, type: str = "HC3") -> dict:
        self._model_is_fitted()
        return robust_se(self, apply, type)

    def variance_inflation_factor(self):
        self._model_is_fitted()
        return variance_inflation_factor(self)
    
    def inference_table(self):
        self._model_is_fitted()
        return inference_table(self)


@dataclass
class LinearRegressionOLS(Model):

    xtx_inv: np.ndarray = field(default=None, repr=False)
    rss: float = None
    tss: float = None
    ess: float = None
    mse: float = None
    rmse: float = None
    f_statistic: float = None
    r_squared: float = None
    r_squared_adjusted: float = None
    t_stat_coefficient: np.ndarray = field(default=None)
    p_value_coefficient: np.ndarray = field(default=None)

    def fit(
        self,
        X:             np.ndarray,
        y:             np.ndarray,
        feature_names: list[str] = None,
        target_name:   str       = None,
        alpha:         float     = 0.05,
    ) -> 'Model':
        
        return fit(self, X, y, feature_names, target_name, alpha)
        

@dataclass
class LogisticRegression(Model):

    xtWx_inv: np.ndarray = field(default=None, repr=False)
    deviance: float = None
    null_deviance: float = None
    pseudo_r_squared: float = None
    lr_statistic: float = None
    z_stat_coefficient: np.ndarray = field(default=None)
    p_value_coefficient: np.ndarray = field(default=None)

    def fit(
        self,
        X:             np.ndarray,
        y:             np.ndarray,
        feature_names: list[str] = None,
        target_name:   str       = None,
        alpha:         float     = 0.05,
        max_iter:      int       = 100,
        tol:           float     = 1e-8,
    ) -> 'Model':

        return fit(self, X, y, feature_names, target_name, alpha, max_iter, tol)
    