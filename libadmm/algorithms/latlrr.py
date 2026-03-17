import numpy as np
from ..proximal_operators import prox_nuclear, prox_l1, prox_l21
from .comp_loss import comp_loss


def latlrr(X, lambda_, opts=None):
    """
    Solve the Latent Low-Rank Representation by M-ADMM

    min_{Z,L,E} ||Z||_*+||L||_*+lambda*loss(E), s.t., XZ+LX-X=E.
    loss(E) = ||E||_1 or 0.5*||E||_F^2 or ||E||_{2,1}

    version 1.0 - 19/06/2016

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

    eta1 = 1.02 * 2 * np.linalg.norm(X, 2) ** 2
    eta2 = eta1
    eta3 = 1.02 * 2

    d, n = X.shape

    E = np.zeros((d, n))
    Z = np.zeros((n, n))
    L = np.zeros((d, d))
    Y = E.copy()

    XtX = X.T @ X
    XXt = X @ X.T

    nuclearnormZ = 0
    nuclearnormL = 0

    for iteration in range(1, max_iter + 1):
        Lk = L.copy()
        Ek = E.copy()
        Zk = Z.copy()

        Z, nuclearnormZ = prox_nuclear(Zk - (X.T @ (Y / mu + L @ X - X - E) + XtX @ Z) / eta1, 1 / (mu * eta1))

        temp = Lk - ((Y / mu + X @ Z - Ek) @ X.T + Lk @ XXt - XXt) / eta2
        L, nuclearnormL = prox_nuclear(temp, 1 / (mu * eta2))

        if loss == 'l1':
            E = prox_l1(Ek + (Y / mu + X @ Z + Lk @ X - X - Ek) / eta3, lambda_ / (mu * eta3))
        elif loss == 'l21':
            E = prox_l21(Ek + (Y / mu + X @ Z + Lk @ X - X - Ek) / eta3, lambda_ / (mu * eta3))
        elif loss == 'l2':
            E = (Y + mu * (X @ Z + Lk @ X - X + (eta3 - 1) * Ek)) / (lambda_ + mu * eta3)
        else:
            raise ValueError('not supported loss function')

        dY = X @ Z + L @ X - X - E

        chgL = np.max(np.abs(Lk - L))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgL, chgE, chgZ, np.max(np.abs(dY))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = nuclearnormZ + nuclearnormL + lambda_ * comp_loss(E, loss)
                err = np.linalg.norm(dY, 'fro') ** 2
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y = Y + mu * dY
        mu = min(rho * mu, max_mu)

    obj = nuclearnormZ + nuclearnormZ + lambda_ * comp_loss(E, loss)
    err = np.linalg.norm(dY, 'fro') ** 2
    return Z, L, obj, err, iteration
