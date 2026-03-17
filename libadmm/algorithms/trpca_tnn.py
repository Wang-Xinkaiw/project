import numpy as np
from ..proximal_operators import prox_tnn, prox_l1


def trpca_tnn(X, lambda_, opts=None):
    """
    Solve the Tensor Robust Principal Component Analysis based on Tensor Nuclear Norm problem by ADMM

    min_{L,S} ||L||_*+lambda*||S||_1, s.t. X=L+S

    version 1.0 - 19/06/2016

    Written by Canyi Lu (canyilu@gmail.com)

    References:
    [1] Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
        Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
        Norm, arXiv preprint arXiv:1804.03728, 2018
    [2] Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
        Yan, Tensor Robust Principal Component Analysis: Exact Recovery of Corrupted
        Low-Rank Tensors via Convex Optimization, arXiv preprint arXiv:1804.03728, 2018
    """
    if opts is None:
        opts = {}

    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    dim = X.shape
    L = np.zeros(dim)
    S = L.copy()
    Y = L.copy()

    tnnL = 0

    for iteration in range(1, max_iter + 1):
        Lk = L.copy()
        Sk = S.copy()

        L, tnnL, _ = prox_tnn(-S + X - Y / mu, 1 / mu)

        S = prox_l1(-L + X - Y / mu, lambda_ / mu)

        dY = L + S - X

        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chg = np.max([chgL, chgS, np.max(np.abs(dY))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = tnnL + lambda_ * np.sum(np.abs(S))
                err = np.linalg.norm(dY)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y = Y + mu * dY
        mu = min(rho * mu, max_mu)

    obj = tnnL + lambda_ * np.sum(np.abs(S))
    err = np.linalg.norm(dY)
    return L, S, obj, err, iteration
