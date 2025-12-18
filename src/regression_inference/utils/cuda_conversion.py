'''
Reassign CUDA objects to numpy arrays
'''
import cupy as cp

def to_numpy(model) -> None:

    model.frozen = False
          
    if isinstance(model.X, cp.ndarray):
        model.X = model.X.get()

    if isinstance(model.coefficients, cp.ndarray):
        model.coefficients = model.coefficients.get()

    if isinstance(model.alpha_cutpoints , cp.ndarray):
        model.alpha_cutpoints = model.alpha_cutpoints.get()

    if isinstance(model.n_classes, cp.ndarray):
        model.n_classes = model.n_classes.get()

    if isinstance(model.xtWx_inv, cp.ndarray):
        model.xtWx_inv = model.xtWx_inv.get()

    model.freeze()
    return model


    