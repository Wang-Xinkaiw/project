import numpy as np
from ..proximal_operators import prox_tnn


def lrtr_Gaussian_tnn(A, b, Xsize, opts=None):
    """
    Low tubal rank tensor recovery from Gaussian measurements by tensor
    nuclear norm minimization

    min_X ||X||_*, s.t. A*vec(X) = b

    version 1.0 - 09/10/2017

    Written by Canyi Lu (canyilu@gmail.com)

    References:
    Canyi Lu, Jiashi Feng, Zhouchen Lin, Shuicheng Yan
    Exact Low Tubal Rank Tensor Recovery from Gaussian Measurements
    International Joint Conference on Artificial Intelligence (IJCAI). 2018
    """
    if opts is None:
        opts = {}

    tol = opts.get('tol', 1e-8)
    max_iter = opts.get('max_iter', 1000)
    rho = opts.get('rho', 1.1)
    mu = opts.get('mu', 1e-6)
    max_mu = opts.get('max_mu', 1e10)
    DEBUG = opts.get('DEBUG', 0)

    n1 = Xsize['n1']
    n2 = Xsize['n2']
    n3 = Xsize['n3']
    X = np.zeros((n1, n2, n3))
    Z = X.copy()
    m = len(b)
    Y1 = np.zeros(m)
    Y2 = X.copy()
    Xtnn = 0

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()
        Zk = Z.copy()

        X, Xtnn, _ = prox_tnn(Z - Y2 / mu, 1 / mu)

        # Calculate vecZ using the same approach as MATLAB
        # Solve (A.T @ A + I) vecZ = rhs
        rhs = A.T @ (-Y1 / mu + b) + Y2.flatten() / mu + X.flatten()
        n = len(rhs)
        
        # For small n, use direct method
        if n < 10000:
            ATA = A.T @ A
            I = np.eye(n)
            vecZ = np.linalg.solve(ATA + I, rhs)
        else:
            # For large n, use iterative method
            # This is a simplified approach
            vecZ = np.linalg.lstsq(A, b - Y1/mu, rcond=None)[0]
            vecZ = vecZ + (Y2.flatten() / mu + X.flatten() - A.T @ vecZ) / (1 + mu)
        Z = vecZ.reshape(n1, n2, n3)

        dY1 = A @ vecZ - b
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = Xtnn
                err = np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = Xtnn
    err = np.linalg.norm(dY1) ** 2 + np.linalg.norm(dY2) ** 2
    return X, obj, err, iteration
