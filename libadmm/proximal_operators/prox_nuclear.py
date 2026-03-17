import numpy as np


def prox_nuclear(B, lambda_):
    """
    The proximal operator of the nuclear norm of a matrix

    min_X lambda*||X||_* + 0.5*||X-B||_F^2

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)
    """
    m, n = B.shape
    U, S, Vh = np.linalg.svd(B, full_matrices=False)
    svp = np.sum(S > lambda_)
    if svp >= 1:
        S_thresh = S[:svp] - lambda_
        X = U[:, :svp] @ np.diag(S_thresh) @ Vh[:svp, :]
        nuclearnorm = np.sum(S_thresh)
    else:
        X = np.zeros((m, n))
        nuclearnorm = 0
    
    return X, nuclearnorm
