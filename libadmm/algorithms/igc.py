import numpy as np
from ..proximal_operators import prox_nuclear, prox_l1, project_box


def igc(A, C, lambda_, opts=None):
    """
    Reference: Chen, Yudong, Sujay Sanghavi, and Huan Xu. Improved graph clustering.

    min_{L,S} ||L||_*+lambda*||C . S||_1, s.t. A=L+S, 0<=L<=1.

    version 1.0 - 19/06/2016

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

    C = np.abs(C)
    d, n = A.shape

    L = np.zeros((d, n))
    S = L.copy()
    Z = L.copy()
    Y1 = L.copy()
    Y2 = L.copy()

    nuclearnormL = 0

    for iteration in range(1, max_iter + 1):
        Lk = L.copy()
        Sk = S.copy()
        Zk = Z.copy()

        L, nuclearnormL = prox_nuclear(Z - Y2 / mu, 1 / mu)

        S = prox_l1(-Z + A - Y1 / mu, C * (lambda_ / mu))

        Z = project_box((-S + A + L + (Y2 - Y1) / mu) / 2, 0, 1)

        dY1 = Z + S - A
        dY2 = L - Z

        chgL = np.max(np.abs(Lk - L))
        chgS = np.max(np.abs(Sk - S))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgL, chgS, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = nuclearnormL + lambda_ * np.sum(C * np.abs(S))
                err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = nuclearnormL + lambda_ * np.sum(C * np.abs(S))
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return L, S, obj, err, iteration
