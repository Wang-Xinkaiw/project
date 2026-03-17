import numpy as np


def tubalrank(X, tol=None):
    """
    The tensor tubal rank of a 3 way tensor

    Parameters:
    -----------
    X : ndarray
        n1*n2*n3 tensor
    tol : float, optional
        tolerance for rank computation

    Returns:
    --------
    trank : int
        tensor tubal rank of X

    version 2.0 - 14/06/2018

    Written by Canyi Lu (canyilu@gmail.com)

    References:
    Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University.
    June, 2018. https://github.com/canyilu/tproduct.

    Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
    Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
    Norm, arXiv preprint arXiv:1804.03728, 2018
    """
    X_fft = np.fft.fft(X, axis=2)
    n1, n2, n3 = X_fft.shape
    s = np.zeros(min(n1, n2))

    s = s + np.linalg.svd(X_fft[:, :, 0], full_matrices=False)[1]

    halfn3 = int(np.round(n3 / 2))
    for i in range(1, halfn3):
        s = s + np.linalg.svd(X_fft[:, :, i], full_matrices=False)[1] * 2

    if n3 % 2 == 0:
        i = halfn3
        s = s + np.linalg.svd(X_fft[:, :, i], full_matrices=False)[1]

    s = s / n3

    if tol is None:
        tol = max(n1, n2) * np.finfo(float).eps * np.max(s)

    trank = np.sum(s > tol)
    return trank
