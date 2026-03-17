import numpy as np
from ..proximal_operators import prox_gl1, prox_l1
from .comp_loss import comp_loss
from .groupl1 import compute_obj


def groupl1R(A, B, G, lambda_, opts=None):
    """
    Solve the group l1 norm regularized minimization problem by M-ADMM

    min_{X,E} loss(E)+lambda*sum_{i=1}^n sum_{g in G} ||(x_i)_g||_2, s.t. AX+E=B

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

    d, na = A.shape
    nb = B.shape[1]

    X = np.zeros((na, nb))
    E = np.zeros((d, nb))
    Z = X.copy()
    Y1 = E.copy()
    Y2 = X.copy()

    AtB = A.T @ B
    I = np.eye(na)
    invAtAI = np.linalg.inv(A.T @ A + I)

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()
        Ek = E.copy()
        Zk = Z.copy()

        for i in range(nb):
            X[:, i] = prox_gl1(Z[:, i] - Y2[:, i] / mu, G, lambda_ / mu)

        if loss == 'l1':
            E = prox_l1(B - A @ Z - Y1 / mu, 1 / mu)
        elif loss == 'l2':
            E = mu * (B - A @ Z - Y1 / mu) / (1 + mu)
        else:
            raise ValueError('not supported loss function')

        Z = invAtAI @ (-A.T @ (Y1 / mu + E) + AtB + Y2 / mu + X)

        dY1 = A @ Z + E - B
        dY2 = X - Z

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgZ = np.max(np.abs(Zk - Z))
        chg = np.max([chgX, chgE, chgZ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = comp_loss(E, loss) + lambda_ * compute_obj(X, G)
                err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        mu = min(rho * mu, max_mu)

    obj = comp_loss(E, loss) + lambda_ * compute_obj(X, G)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, E, obj, err, iteration
