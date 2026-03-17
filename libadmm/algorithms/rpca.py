import numpy as np
from ..proximal_operators import prox_nuclear, prox_l1, prox_l21
from .comp_loss import comp_loss


def rpca(X, lambda_, opts=None):
    """
    Solve the Robust Principal Component Analysis minimization problem by M-ADMM

    min_{L,S} ||L||_*+lambda*loss(S), s.t. X=L+S
    loss(S) = ||S||_1 or ||S||_{2,1}

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

    d, n = X.shape

    L = np.zeros((d, n))
    S = L.copy()
    Y = L.copy()

    nuclearnormL = 0

    for iteration in range(1, max_iter + 1):
        Lk = L.copy()
        Sk = S.copy()

        L, nuclearnormL = prox_nuclear(-S + X - Y / mu, 1 / mu)

        if loss == 'l1':
            S = prox_l1(-L + X - Y / mu, lambda_ / mu)
        elif loss == 'l21':
            S = prox_l21(-L + X - Y / mu, lambda_ / mu)
        else:
            raise ValueError('not supported loss function')

        dY = L + S - X

        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chg = np.max([chgL, chgS, np.max(np.abs(dY))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = nuclearnormL + lambda_ * comp_loss(S, loss)
                err = np.linalg.norm(dY, 'fro')
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y = Y + mu * dY
        mu = min(rho * mu, max_mu)

    obj = nuclearnormL + lambda_ * comp_loss(S, loss)
    err = np.linalg.norm(dY, 'fro')
    return L, S, obj, err, iteration
