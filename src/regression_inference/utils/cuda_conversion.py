import cupy as cp

'''
to_numpy(model: Model = self)

    Reassign CUDA objects to numpy arrays

    CUDA Fit functions call 'to_numpy' after the fit is complete,
    and before the post-fit freeze.

'''

def to_numpy(model) -> None:

    for attr in vars(model):
        value = getattr(model, attr)

        if isinstance(value, cp.ndarray):
            setattr(model, attr, value.get())
        
    return model


    