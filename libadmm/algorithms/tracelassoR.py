import numpy as np
from ..proximal_operators import prox_nuclear, prox_l1
from .comp_loss import comp_loss
from .tracelasso import _Adiagx, _diagAtB


def tracelassoR(A, b, lambda_, opts=None):
    """
    Solve the trace Lasso regularized minimization problem by M-ADMM

    min_{x,e} loss(e)+lambda*||A*Diag(x)||_*, s.t. Ax+e=b
    loss(e) = ||e||_1 or 0.5*||e||_2^2

    version 1.0 - 18/06/2016

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

    d, n = A.shape

    x = np.zeros(n)
    Z = np.zeros((d, n))
    e = np.zeros(d)
    Y1 = e.copy()
    Y2 = Z.copy()

    Atb = A.T @ b
    AtA = A.T @ A
    invAtA = np.linalg.inv(AtA + np.diag(np.diag(AtA)))

    nuclearnorm = 0

    for iteration in range(1, max_iter + 1):
        xk = x.copy()
        ek = e.copy()
        Zk = Z.copy()

        Z, nuclearnorm = prox_nuclear(_Adiagx(A, x) - Y2 / mu, lambda_ / mu)

        if loss == 'l1':
            e = prox_l1(b - A @ x - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            e = mu * (b - A @ x - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('not supported loss function')

        x = invAtA @ (-A.T @ (Y1 / mu + e) + Atb + _diagAtB(A, Y2 / mu + Z))

        dY1 = A @ x + e - b
        dY2 = Z - _Adiagx(A, x)

        chgx = np.max(np.abs(xk - x))
        chge = np.max(np.abs(ek - e))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgx, chge, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = comp_loss(e, loss) + lambda_ * nuclearnorm
                err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = comp_loss(e, loss) + lambda_ * nuclearnorm
    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return x, e, obj, err, iteration
