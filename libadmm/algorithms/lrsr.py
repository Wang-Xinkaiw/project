import numpy as np
from ..proximal_operators import prox_nuclear, prox_l1, prox_l21
from .comp_loss import comp_loss


def lrsr(A, B, lambda1, lambda2, opts=None):
    """
    Solve the Low-Rank and Sparse Representation (LRSR) minimization problem by M-ADMM

    min_{X,E} ||X||_*+lambda1*||X||_1+lambda2*loss(E), s.t. A=BX+E
    loss(E) = ||E||_1 or 0.5*||E||_F^2 or ||E||_{2,1}

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)
    """
    if opts is None:
        opts = {}

    loss = opts.get('loss', 'l21')
    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 500)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-4)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    d, na = A.shape
    nb = B.shape[1]

    X = np.zeros((nb, na))
    E = np.zeros((d, na))
    Z = X.copy()
    J = X.copy()

    Y1 = E.copy()
    Y2 = X.copy()
    Y3 = X.copy()
    BtB = B.T @ B
    BtA = B.T @ A
    I = np.eye(nb)
    invBtBI = np.linalg.inv(BtB + 2 * I)

    nuclearnormZ = 0

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()
        Ek = E.copy()
        Jk = J.copy()

        Z, nuclearnormZ = prox_nuclear(X + Y2 / mu, 1 / mu)
        J = prox_l1(X + Y3 / mu, lambda1 / mu)

        if loss == 'l1':
            E = prox_l1(A - B @ X + Y1 / mu, lambda2 / mu)
        elif loss == 'l21':
            E = prox_l21(A - B @ X + Y1 / mu, lambda2 / mu)
        elif loss == 'l2':
            E = mu * (A - B @ X + Y1 / mu) / (lambda2 + mu)
        else:
            raise ValueError('not supported loss function')

        X = invBtBI @ (B.T @ (Y1 / mu - E) + BtA - (Y2 + Y3) / mu + Z + J)

        dY1 = A - B @ X - E
        dY2 = X - Z
        dY3 = X - J

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chgJ = np.max(np.abs(Jk - J))
        chg = np.max([chgX, chgE, chgZ, chgJ, np.max(np.abs(dY1)), np.max(np.abs(dY2)), np.max(np.abs(dY3))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = nuclearnormZ + lambda1 * np.sum(np.abs(J)) + lambda2 * comp_loss(E, loss)
                err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2 + np.linalg.norm(dY3, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = nuclearnormZ + lambda1 * np.sum(np.abs(J)) + lambda2 * comp_loss(E, loss)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2 + np.linalg.norm(dY3, 'fro') ** 2)
    return X, E, obj, err, iteration
