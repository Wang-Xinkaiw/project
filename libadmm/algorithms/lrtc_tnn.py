import numpy as np
from ..proximal_operators import prox_tnn


def lrtc_tnn(M, omega, opts=None):
    """
    Solve the Low-Rank Tensor Completion (LRTC) based on Tensor Nuclear Norm (TNN) problem by M-ADMM

    min_X ||X||_*, s.t. P_Omega(X) = P_Omega(M)

    version 1.0 - 25/06/2016

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

    E = np.zeros(dim)
    Y = E.copy()

    tnnX = 0

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()
        Ek = E.copy()

        X, tnnX, _ = prox_tnn(-E + M + Y / mu, 1 / mu)

        E = M - X + Y / mu
        E_flat = E.flatten()
        E_flat[omega] = 0
        E = E_flat.reshape(dim)

        dY = M - X - E

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chg = np.max([chgX, chgE, np.max(np.abs(dY))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = tnnX
                err = np.linalg.norm(dY)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y = Y + mu * dY
        mu = min(rho * mu, max_mu)

    obj = tnnX
    err = np.linalg.norm(dY)
    return X, obj, err, iteration
