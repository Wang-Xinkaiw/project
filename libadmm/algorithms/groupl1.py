import numpy as np
from ..proximal_operators import prox_gl1


def compute_obj(X, G):
    """
    Compute group l1 objective
    """
    obj = 0
    for i in range(X.shape[1]):
        x = X[:, i]
        for j in range(len(G)):
            obj = obj + np.linalg.norm(x[G[j]])
    return obj


def groupl1(A, B, G, opts=None):
    """
    Solve the group l1-minimization problem by ADMM

    min_X sum_{i=1}^n sum_{g in G} ||(x_i)_g||_2, s.t. AX=B

    Parameters:
    -----------
    A : ndarray
        d*na matrix
    B : ndarray
        d*nb matrix
    G : list
        a list indicates a partition of 1:na
    opts : dict, optional
        Structure value in Python.

    Returns:
    --------
    X : ndarray
        na*nb matrix
    obj : float
        objective function value
    err : float
        residual ||AX-B||_F
    iter : int
        number of iterations

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

        for i in range(nb):
            X[:, i] = prox_gl1(Z[:, i] - Y2[:, i] / mu, G, 1 / mu)

        Z = invAtAI @ (-(A.T @ Y1 - Y2) / mu + AtB + X)

        dY1 = A @ Z - B
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = compute_obj(X, G)
                err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = compute_obj(X, G)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, obj, err, iteration
