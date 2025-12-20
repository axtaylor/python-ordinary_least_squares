'''
Reassign CUDA objects to numpy arrays

freeze = False 

    - When applying robust covariance pre-fit

freeze = True (default)

    - Any conversions done post fit.

'''
import cupy as cp

def to_numpy(model, freeze=True) -> None:

    model.frozen = False

    for attr in vars(model):

        value = getattr(model, attr)

        if isinstance(value, cp.ndarray):

            setattr(model, attr, value.get())

    if freeze:
        model.freeze()
        
    return model


    