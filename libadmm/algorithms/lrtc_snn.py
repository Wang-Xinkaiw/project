import numpy as np
from ..proximal_operators import prox_nuclear
from ..tensor_tools import Fold, Unfold


def lrtc_snn(M, omega, alpha, opts=None):
    """
    Solve the Low-Rank Tensor Completion (LRTC) based on Sum of Nuclear Norm (SNN) problem by M-ADMM

    min_X sum_i alpha_i*||X_{i(i)}||_*, s.t. P_Omega(X) = P_Omega(M)

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

    dim = np.array(M.shape)
    k = len(dim)

    omega = np.asarray(omega).flatten()
    omegac = np.setdiff1d(np.arange(np.prod(dim)), omega)

    X = np.zeros(dim)
    X_flat = X.flatten()
    M_flat = M.flatten()
    X_flat[omega] = M_flat[omega]
    X = X_flat.reshape(dim)

    Y = np.zeros(dim)

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()

        # Update X by solving proximal for each mode
        X_sum = np.zeros(dim)
        for i in range(k):
            X_unfold = Unfold(X - Y / mu, dim, i + 1)
            Z_mode, _ = prox_nuclear(X_unfold, alpha[i] / mu)
            Z_mode = Fold(Z_mode, dim, i + 1)
            X_sum = X_sum + Z_mode
        X = X_sum / k

        # Project to observed entries
        X_flat = X.flatten()
        M_flat = M.flatten()
        X_flat[omega] = M_flat[omega]
        X = X_flat.reshape(dim)

        # Update Y
        dY = X - Xk
        Y = Y + mu * dY

        # Calculate change
        chg = np.max(np.abs(dY))
        err_val = np.linalg.norm(dY)

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                print(f'iter {iteration}, mu={mu}, err={err_val}')

        if chg < tol:
            break

        mu = min(rho * mu, max_mu)

    return X, err_val, iteration
