import numpy as np
from ..proximal_operators import prox_l1
from .comp_loss import comp_loss


def comp_fusedl1(x, lambda1, lambda2):
    """
    Compute fused l1 norm
    f = lambda1*||x||_1 + lambda2*sum_{i=2}^p |x_i-x_{i-1}|
    """
    f = 0
    p = len(x)
    for i in range(1, p):
        f = f + np.abs(x[i] - x[i - 1])
    f = lambda1 * np.sum(np.abs(x)) + lambda2 * f
    return f


def prox_fusedl1(v, lambda1, lambda2, n):
    """
    Proximal operator for fused l1 norm
    """
    x = np.zeros_like(v)
    for i in range(n):
        if i == 0:
            x[i] = np.sign(v[i]) * max(abs(v[i]) - lambda1, 0)
        else:
            diff = v[i] - x[i - 1]
            thresh = np.sign(diff) * max(abs(diff) - lambda2, 0)
            x[i] = x[i - 1] + thresh
            val = v[i] - lambda1
            if abs(x[i]) > abs(val):
                x[i] = val
            x[i] = np.sign(x[i]) * max(abs(x[i]) - lambda1, 0)
    return x


def fusedl1R(A, b, lambda1, lambda2, opts=None):
    """
    Solve the fused Lasso regularized minimization problem by ADMM

    min_{x,e} loss(e) + lambda1*||x||_1 + lambda2*sum_{i=2}^p |x_i-x_{i-1}|,
    loss(e) = ||e||_1 or 0.5*||e||_2^2

    Parameters:
    -----------
    A : ndarray
        d*n matrix
    b : ndarray
        d*1 vector
    lambda1 : float
        >=0, parameter
    lambda2 : float
        >=0, parameter
    opts : dict, optional
        Structure value in Python. The fields are:
            opts.loss       -   'l1' (default): loss(e) = ||e||_1
                                'l2': loss(E) = 0.5*||e||_2^2
            opts.tol        -   termination tolerance
            opts.max_iter   -   maximum number of iterations
            opts.mu         -   stepsize for dual variable updating in ADMM
            opts.max_mu     -   maximum stepsize
            opts.rho        -   rho>=1, ratio used to increase mu
            opts.DEBUG      -   0 or 1

    Returns:
    --------
    x : ndarray
        n*1 vector
    e : ndarray
        d*1 vector
    obj : float
        objective function value
    err : float
        residual
    iter : int
        number of iterations

    version 1.0 - 20/06/2016

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
    e = np.zeros(d)
    z = x.copy()
    Y1 = e.copy()
    Y2 = x.copy()

    Atb = A.T @ b
    I = np.eye(n)
    invAtAI = np.linalg.inv(A.T @ A + I)

    tol2 = 1e-10
    max_step = 50
    x0 = np.zeros(n - 1)

    for iteration in range(1, max_iter + 1):
        xk = x.copy()
        ek = e.copy()
        zk = z.copy()

        x = prox_fusedl1(z - Y2 / mu, lambda1 / mu, lambda2 / mu, n)

        if loss == 'l1':
            e = prox_l1(b - A @ z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            e = mu * (b - A @ z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('not supported loss function')

        z = invAtAI @ (-A.T @ (Y1 / mu + e) + Atb + Y2 / mu + x)

        dY1 = A @ z + e - b
        dY2 = x - z

        chgx = np.max(np.abs(xk - x))
        chge = np.max(np.abs(ek - e))
        chgz = np.max(np.abs(zk - z))
        chg = np.max([chgx, chge, chgz, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = comp_loss(e, loss) + comp_fusedl1(x, lambda1, lambda2)
                err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = comp_loss(e, loss) + comp_fusedl1(x, lambda1, lambda2)
    err = np.sqrt(np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2)
    return x, e, obj, err, iteration
