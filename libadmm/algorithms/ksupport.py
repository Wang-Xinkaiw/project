import numpy as np
from ..proximal_operators import prox_ksupport


def ksupport(A, B, k, opts=None):
    """
    Solve the k support norm minimization problem by ADMM

    min_X 0.5*||vec(X)||_ksp^2, s.t. AX=B

    version 1.0 - 27/06/2016

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

    d, na = A.shape
    nb = B.shape[1]

    X = np.zeros((na, nb))
    Z = X.copy()
    Y1 = np.zeros((d, nb))
    Y2 = X.copy()

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()

        temp = Z - Y2 / mu
        temp = prox_ksupport(temp.flatten(), k, 1 / mu)
        X = temp.reshape(na, nb)

        Z = invAtAI @ (-A.T @ Y1 / mu + AtB + Y2 / mu + X)

        dY1 = A @ Z - B
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, err, iteration
