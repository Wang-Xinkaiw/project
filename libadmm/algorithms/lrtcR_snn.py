import numpy as np
from ..proximal_operators import prox_nuclear, prox_l1
from ..tensor_tools import Fold, Unfold


def lrtcR_snn(M, omega, alpha, opts=None):
    """
    Solve the Noisy Low-Rank Tensor Completion (LRTC) based on Sum of Nuclear Norm (SNN) problem by M-ADMM

    min_{X,E} sum_i alpha_i*||X_{i(i)}||_* + loss(E), s.t. P_Omega(X) + E = M.
    loss(E) = ||E||_1 or 0.5*||E||_F^2

    version 1.0 - 24/06/2016

    Written by Canyi Lu (canyilu@gmail.com)
    """
    if opts is None:
        opts = {}

    loss = opts.get('loss', 'l1')
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
    E = np.zeros(dim)
    Y = np.zeros(dim)

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()
        Ek = E.copy()

        # Update X by solving proximal for each mode
        X_sum = np.zeros(dim)
        for i in range(k):
            X_unfold = Unfold(X - Y / mu, dim, i + 1)
            Z_mode, _ = prox_nuclear(X_unfold, alpha[i] / mu)
            Z_mode = Fold(Z_mode, dim, i + 1)
            X_sum = X_sum + Z_mode
        X = X_sum / k

        # Update E
        if loss == 'l1':
            E = prox_l1(M - X, 1 / mu)
        elif loss == 'l2':
            E = (M - X) * (mu / (1 + mu))
        else:
            raise ValueError('not supported loss function')

        # Project X to observed entries
        X_flat = X.flatten()
        M_flat = M.flatten()
        E_flat = E.flatten()
        X_flat[omega] = M_flat[omega] - E_flat[omega]
        X = X_flat.reshape(dim)

        # Update Y
        dY = X + E - M
        Y = Y + mu * dY

        # Calculate change
        chg = np.max([np.max(np.abs(Xk - X)), np.max(np.abs(Ek - E)), np.max(np.abs(dY))])
        err_val = np.linalg.norm(dY)

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                print(f'iter {iteration}, mu={mu}, err={err_val}')

        if chg < tol:
            break

        mu = min(rho * mu, max_mu)

    return X, err_val, iteration
