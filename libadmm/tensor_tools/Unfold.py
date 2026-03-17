import numpy as np


def Unfold(X, dim, i):
    """
    Unfold a tensor along a specific mode

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)

    Parameters:
    -----------
    X : ndarray
        tensor to unfold
    dim : tuple
        dimensions of the tensor
    i : int
        mode along which to unfold (1-indexed)

    Returns:
    --------
    X : ndarray
        unfolded matrix of size dim(i) x prod(dim)/dim(i)
    """
    X = np.moveaxis(X, i - 1, 0)
    X = X.reshape(dim[i - 1], -1)
    return X
