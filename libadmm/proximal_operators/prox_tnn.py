import numpy as np


def prox_tnn(Y, rho):
    """
    The proximal operator of the tensor nuclear norm of a 3 way tensor

    min_X rho*||X||_* + 0.5*||X-Y||_F^2

    Parameters:
    -----------
    Y : ndarray
        n1*n2*n3 tensor
    rho : float
        regularization parameter

    Returns:
    --------
    X : ndarray
        n1*n2*n3 tensor
    tnn : float
        tensor nuclear norm of X
    trank : int
        tensor tubal rank of X

    version 2.1 - 14/06/2018

    Written by Canyi Lu (canyilu@gmail.com)

    References:
    Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University.
    June, 2018. https://github.com/canyilu/tproduct.

    Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
    Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
    Norm, arXiv preprint arXiv:1804.03728, 2018
    """
    n1, n2, n3 = Y.shape
    X = np.zeros((n1, n2, n3))
    Y_fft = np.fft.fft(Y, axis=2)
    tnn = 0
    trank = 0

    U, S, Vh = np.linalg.svd(Y_fft[:, :, 0], full_matrices=False)
    r = np.sum(S > rho)
    if r >= 1:
        S = S[:r] - rho
        X[:, :, 0] = U[:, :r] @ np.diag(S) @ Vh[:r, :]
        tnn = tnn + np.sum(S)
        trank = max(trank, r)

    halfn3 = int(np.floor(n3 / 2))
    for i in range(1, halfn3 + 1):
        U, S, Vh = np.linalg.svd(Y_fft[:, :, i], full_matrices=False)
        r = np.sum(S > rho)
        if r >= 1:
            S = S[:r] - rho
            X[:, :, i] = U[:, :r] @ np.diag(S) @ Vh[:r, :]
            tnn = tnn + np.sum(S) * 2
            trank = max(trank, r)
        if i < n3 - i:
            X[:, :, n3 - i] = np.conj(X[:, :, i])

    if n3 % 2 == 1 or n3 == 2:
        i = halfn3 if n3 % 2 == 0 else halfn3
        if i < n3 and i > 0:
            U, S, Vh = np.linalg.svd(Y_fft[:, :, i], full_matrices=False)
            r = np.sum(S > rho)
            if r >= 1:
                S = S[:r] - rho
                X[:, :, i] = U[:, :r] @ np.diag(S) @ Vh[:r, :]
                tnn = tnn + np.sum(S)
                trank = max(trank, r)

    tnn = tnn / n3
    X = np.fft.ifft(X, axis=2)
    if np.iscomplexobj(X):
        X = np.real(X)
    return X, tnn, trank
