'''
Reassign CUDA objects to numpy arrays
'''
import cupy as cp

def to_numpy(model, freeze=True) -> None:

    model.frozen = False
          
    if isinstance(model.X, cp.ndarray):
        model.X = model.X.get()

    if isinstance(model.coefficients, cp.ndarray):
        model.coefficients = model.coefficients.get()
    
    if isinstance(model.intercept, cp.ndarray):
        model.intercept = model.intercept.get()

    if isinstance(model.theta, cp.ndarray):
        model.theta = model.theta.get()

    if hasattr(model, 'alpha_cutpoints'):
        if isinstance(model.alpha_cutpoints , cp.ndarray):
            model.alpha_cutpoints = model.alpha_cutpoints.get()

    if hasattr(model, 'y_encoded'):
        if isinstance(model.y_encoded , cp.ndarray):
            model.y_encoded = model.y_encoded.get()

    if hasattr(model, 'probabilities'):
        if isinstance(model.probabilities , cp.ndarray):
            model.probabilities = model.probabilities.get()

    if isinstance(model.n_classes, cp.ndarray):
        model.n_classes = model.n_classes.get()

    if isinstance(model.xtWx_inv, cp.ndarray):
        model.xtWx_inv = model.xtWx_inv.get()

    if freeze:
        model.freeze()
        
    return model


    