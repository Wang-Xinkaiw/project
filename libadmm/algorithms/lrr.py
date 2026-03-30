import numpy as np
from ..proximal_operators import prox_nuclear, prox_l1, prox_l21
from .comp_loss import comp_loss


def lrr(A, B, lambda_, opts=None, strategy=None):
    """
    Solve the Low-Rank Representation minimization problem by M-ADMM

    min_{X,E} ||X||_*+lambda*loss(E), s.t. A=BX+E
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
    J = X.copy()

    Y1 = E.copy()
    Y2 = X.copy()
    BtB = B.T @ B
    BtA = B.T @ A
    I = np.eye(nb)
    invBtBI = np.linalg.inv(BtB + I)

    nuclearnormJ = 0

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()
        Ek = E.copy()
        Jk = J.copy()

        J, nuclearnormJ = prox_nuclear(X + Y2 / mu, 1 / mu)

        if loss == 'l1':
            E = prox_l1(A - B @ X + Y1 / mu, lambda_ / mu)
        elif loss == 'l21':
            E = prox_l21(A - B @ X + Y1 / mu, lambda_ / mu)
        elif loss == 'l2':
            E = mu * (A - B @ X + Y1 / mu) / (lambda_ + mu)
        else:
            raise ValueError('not supported loss function')

        X = invBtBI @ (B.T @ (Y1 / mu - E) + BtA - Y2 / mu + J)

        dY1 = A - B @ X - E
        dY2 = X - J

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chgJ = np.max(np.abs(Jk - J))
        chg = np.max([chgX, chgE, chgJ, np.max(np.abs(dY1)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = nuclearnormJ + lambda_ * comp_loss(E, loss)
                err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y1 = Y1 + mu * dY1
        Y2 = Y2 + mu * dY2
        # 使用策略更新 mu（如果提供了 strategy）

        if strategy is not None:

            try:

                iteration_state = {

                    'iteration': iteration,

                    'primal_residual': float(np.linalg.norm(dY1, 'fro')),

                    'dual_residual': float(np.linalg.norm(dY2, 'fro')),

                    'beta': mu,

                    'objective': float(obj_temp),

                    'converged': False

                }

                strategy_update = strategy.update_parameters(iteration_state)

                if 'beta' in strategy_update:

                    mu = float(strategy_update['beta'])

                    mu = min(max(mu, 1e-10), max_mu)

            except Exception:

                mu = min(rho * mu, max_mu)

        else:

            mu = min(rho * mu, max_mu)

    obj = nuclearnormJ + lambda_ * comp_loss(E, loss)
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, E, obj, err, iteration
