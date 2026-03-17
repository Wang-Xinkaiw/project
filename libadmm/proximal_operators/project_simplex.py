import numpy as np


def project_simplex(B):
    """
    Project onto the probability simplex
    min_X ||X-B||_F
    s.t Xe=e, X>=0 where e is the constant one vector.

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)

    Parameters:
    -----------
    B : ndarray
        n*d matrix

    Returns:
    --------
    X : ndarray
        n*d matrix
    """
    n, m = B.shape
    A = np.tile(np.arange(1, m + 1), (n, 1))
    B_sort = np.sort(B, axis=1, kind='quicksort')[:, ::-1]
    cum_B = np.cumsum(B_sort, axis=1)
    sigma = B_sort - (cum_B - 1) / A
    tmp = sigma > 0
    idx = np.sum(tmp, axis=1)
    tmp = B_sort - sigma
    sigma_diag = tmp[np.arange(n), idx - 1]
    sigma = np.tile(sigma_diag.reshape(-1, 1), (1, m))
    X = np.maximum(B - sigma, 0)
    return X
