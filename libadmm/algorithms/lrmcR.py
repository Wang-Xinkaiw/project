import numpy as np
from ..proximal_operators import prox_nuclear, prox_l1, prox_l21
from .comp_loss import comp_loss


def lrmcR(M, omega, lambda_, opts=None):
    """
    Solve the Noisy Low-Rank Matrix Completion (LRMC) problem by ADMM

    min_{X,E} ||X||_*+lambda*loss(E), s.t. P_Omega(X) + E = M.
    loss(E) = ||E||_1 or 0.5*||E||_F^2 or ||E||_{2,1}

    version 1.0 - 23/06/2016

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

    d, n = M.shape
    X = np.zeros((d, n))
    Z = X.copy()
    E = X.copy()
    Y1 = X.copy()
    Y2 = X.copy()

    # Convert tuple indices (from np.where) to linear indices
    if isinstance(omega, tuple) and len(omega) == 2:
        rows, cols = omega
        d, n = M.shape
        omega = rows * n + cols
    else:
        omega = np.asarray(omega).flatten()
    omegac = np.setdiff1d(np.arange(d * n), omega)

    nuclearnormX = 0

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        Ek = E.copy()

        X, nuclearnormX = prox_nuclear(Z - Y2 / mu, 1 / mu)

        temp = M - Y1 / mu
        temp_flat = temp.flatten()
        temp_flat[omega] = temp_flat[omega] - Z.flatten()[omega]
        temp = temp_flat.reshape(d, n)

        if loss == 'l1':
            E = prox_l1(temp, lambda_ / mu)
        elif loss == 'l21':
            E = prox_l21(temp, lambda_ / mu)
        elif loss == 'l2':
            E = temp * (mu / (lambda_ + mu))
        else:
            raise ValueError('not supported loss function')

        Z_flat = Z.flatten()
        X_flat = X.flatten()
        E_flat = E.flatten()
        M_flat = M.flatten()
        Y1_flat = Y1.flatten()
        Y2_flat = Y2.flatten()

        Z_flat[omega] = (-E_flat[omega] + M_flat[omega] - (Y1_flat[omega] - Y2_flat[omega]) / mu + X_flat[omega]) / 2
        Z_flat[omegac] = X_flat[omegac] + Y2_flat[omegac] / mu
        Z = Z_flat.reshape(d, n)

        dY1 = E - M
        dY1_flat = dY1.flatten()
        dY1_flat[omega] = dY1_flat[omega] + Z_flat[omega]
        dY1 = dY1_flat.reshape(d, n)
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = nuclearnormX + lambda_ * comp_loss(E, loss)
                err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = nuclearnormX + lambda_ * comp_loss(E, loss)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, E, obj, err, iteration
