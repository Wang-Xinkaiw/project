import numpy as np
from ..proximal_operators import prox_nuclear


def _Adiagx(A, x):
    """Compute A*Diag(x) = A * diag(x)"""
    return A * x


def _diagAtB(A, B):
    """Compute diag(A'*B)"""
    n = A.shape[1]
    v = np.zeros(n)
    for i in range(n):
        v[i] = A[:, i].T @ B[:, i]
    return v


def tracelasso(A, b, opts=None):
    """
    Solve the trace Lasso minimization problem by ADMM

    min_x ||A*Diag(x)||_*, s.t. Ax=b

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

    d, n = A.shape

    x = np.zeros(n)
    Z = np.zeros((d, n))
    Y1 = np.zeros(d)
    Y2 = Z.copy()

    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = np.linalg.inv(AtA + np.diag(np.diag(AtA)))

    nuclearnorm = 0

    for iteration in range(1, max_iter + 1):
        xk = x.copy()
        Zk = Z.copy()

        x = invAtA @ (-A.T @ Y1 / mu + Atb + _diagAtB(A, -Y2 / mu + Z))

        Z, nuclearnorm = prox_nuclear(_Adiagx(A, x) + Y2 / mu, 1 / mu)

        dY1 = A @ x - b
        dY2 = _Adiagx(A, x) - Z

        chgx = np.max(np.abs(xk - x))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgx, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = nuclearnorm
                err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = nuclearnorm
    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return x, obj, err, iteration
