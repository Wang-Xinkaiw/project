import numpy as np


def prox_gl1(b, G, lambda_):
    """
    The proximal operator of the group l1 norm

    min_x lambda*sum_{g in G} ||x_g||_2 + 0.5*||x-b||_2^2

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)

    Parameters:
    -----------
    b : ndarray
        d*1 vector
    G : list
        a list indicates a partition of 1:d
    lambda_ : float
        regularization parameter

    Returns:
    --------
    x : ndarray
        d*1 vector
    """
    x = np.zeros_like(b)
    for i in range(len(G)):
        nxg = np.linalg.norm(b[G[i]])
        if nxg > lambda_:
            x[G[i]] = b[G[i]] * (1 - lambda_ / nxg)
    return x
