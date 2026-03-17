import numpy as np


def Fold(X, dim, i):
    """
    Fold a tensor along a specific mode

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)
    
    MATLAB equivalent:
    function [X] = Fold(X, dim, i)
        dim = circshift(dim, [1-i, 1-i]);
        X = shiftdim(reshape(X, dim), length(dim)+1-i);
    """
    # Shift dimensions
    dim_shifted = np.roll(dim, 1 - i)
    # Reshape
    X = X.reshape(dim_shifted)
    # Shift dimensions back (shiftdim with negative shifts moves dimensions to the end)
    shift = len(dim) + 1 - i
    X = np.moveaxis(X, list(range(shift)), list(range(len(dim) - shift, len(dim))))
    return X
