import numpy as np   
from ..models.predict import predict_linear, predict_logit, predict_logit_multinomial, predict_logit_ordinal 
from typing import Union

def predict(model, X: np.ndarray, alpha: float, return_table: bool) -> Union[float, np.ndarray, dict, list[dict]]:

    X = np.asarray(X, dtype=float)

    if model.model_type == "linear":
        return predict_linear.predict(model, X, alpha, return_table)               # Returns: Union[float, dict]

    elif model.model_type == "logit":
        return predict_logit.predict(model, X, alpha, return_table)                # Returns: Union[np.ndarray, dict]

    elif model.model_type == "logit_multinomial":
        return predict_logit_multinomial.predict(model, X, alpha, return_table)    # Returns: Union[np.ndarray, dict]

    elif model.model_type == "logit_ordinal":
        return predict_logit_ordinal.predict(model, X, alpha, return_table)        # Returns: Union[np.ndarray, List[dict]]

    else:
        raise ValueError(f"Model type: {model.model_type} is unexpected.")


