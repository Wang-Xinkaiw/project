import numpy as np
from ..proximal_operators import prox_nuclear, prox_l1
from ..tensor_tools import Fold, Unfold


def trpca_snn(X, alpha, opts=None):
    """
    Solve the Tensor Robust Principal Component Analysis (TRPCA) based on Sum of Nuclear Norm (SNN) problem by M-ADMM

    min_{L,E} sum_i alpha_i*||L_{i(i)}||_* + ||E||_1, s.t. X = L + E.

    version 1.0 - 24/06/2016

    Written by Canyi Lu (canyilu@gmail.com)
    """
    if opts is None:
        opts = {}

    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    dim = np.array(X.shape)
    k = len(dim)

    E = np.zeros(dim)
    Y = np.zeros(dim)
    L = np.zeros(dim)

    for iteration in range(1, max_iter + 1):
        Lk = L.copy()
        Ek = E.copy()

        # Update L by solving proximal for each mode
        L_sum = np.zeros(dim)
        for i in range(k):
            X_unfold = Unfold(X - E - Y / mu, dim, i + 1)
            L_mode, _ = prox_nuclear(X_unfold, alpha[i] / mu)
            L_mode = Fold(L_mode, dim, i + 1)
            L_sum = L_sum + L_mode
        L = L_sum / k

        # Update E
        E = prox_l1(X - L - Y / mu, 1 / mu)

        # Update Y
        dY = L + E - X
        Y = Y + mu * dY

        # Calculate change
        chg = np.max([np.max(np.abs(Lk - L)), np.max(np.abs(Ek - E)), np.max(np.abs(dY))])
        err_val = np.linalg.norm(dY)

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                print(f'iter {iteration}, mu={mu}, err={err_val}')

        if chg < tol:
            break

        mu = min(rho * mu, max_mu)

    return L, E, err_val, iteration
