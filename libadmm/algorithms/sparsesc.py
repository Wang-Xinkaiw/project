import numpy as np
from ..proximal_operators import prox_l1, project_fantope


def sparsesc(L, lambda_, k, opts=None):
    """
    Solve the Sparse Spectral Clustering problem

    min_P <P,L>+lambda*||P||_1, s.t. 0<=P<=I, Tr(P)=k

    Reference: Canyi Lu, Shuicheng Yan, Zhouchen Lin, Convex Sparse Spectral
    Clustering: Single-view to Multi-view, TIP, 2016

    version 1.0 - 18/06/2016

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

    n = L.shape[0]
    P = np.zeros((n, n))
    Q = P.copy()
    Y = P.copy()

    for iteration in range(1, max_iter + 1):
        Pk = P.copy()
        Qk = Q.copy()

        P = prox_l1(Q - (Y + L) / mu, lambda_ / mu)

        temp = P + Y / mu
        temp = (temp + temp.T) / 2
        Q = project_fantope(temp, k)

        dY = P - Q

        chgP = np.max(np.abs(Pk - P))
        chgQ = np.max(np.abs(Qk - Q))
        chg = np.max([chgP, chgQ, np.max(np.abs(dY))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = np.trace(P.T @ L) + lambda_ * np.sum(np.abs(Q))
                err = np.linalg.norm(dY, 'fro')
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y = Y + mu * dY
        mu = min(rho * mu, max_mu)

    obj = np.trace(P.T @ L) + lambda_ * np.sum(np.abs(Q))
    err = np.linalg.norm(dY, 'fro')
    return P, obj, err, iteration
