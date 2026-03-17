import numpy as np


def _cappedsimplexprojection(v, k):
    """
    Project onto the capped simplex
    min_x ||x-v||_2, s.t. 0 <= x <= 1, sum(x) = k
    """
    n = len(v)
    v_sort = np.sort(v)[::-1]
    cum_v = np.cumsum(v_sort)
    sigma = v_sort - (cum_v - 1) / np.arange(1, n + 1)
    tmp = sigma > 0
    idx = np.sum(tmp)
    if idx == 0:
        theta = 0
    else:
        theta = (cum_v[idx - 1] - k) / idx if idx > 0 else 0
    x = np.maximum(v - theta, 0)
    x = np.minimum(x, 1)
    return x


def project_fantope(Q, k):
    """
    Project a point onto the Fantope
    Q - a symmetric matrix

    min_X ||X-Q||_F, s.t. 0 >= X >= I, Tr(X)=k.

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)
    """
    D, U = np.linalg.eigh(Q)
    Dr = _cappedsimplexprojection(D, k)
    X = U @ np.diag(Dr) @ U.T
    return X
