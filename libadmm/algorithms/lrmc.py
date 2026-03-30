import numpy as np
from ..proximal_operators import prox_nuclear


def lrmc(MM, omega, opts=None, strategy=None):
    """
    Solve the Low-Rank Matrix Completion (LRMC) problem by ADMM

    min_X ||X||_*, s.t. P_Omega(X) = P_Omega(M)

    version 1.0 - 22/06/2016

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

    d, n = MM.shape

    # Convert tuple indices (from np.where) to linear indices
    if isinstance(omega, tuple) and len(omega) == 2:
        rows, cols = omega
        d, n = MM.shape
        omega = rows * n + cols
    else:
        omega = np.asarray(omega).flatten()

    M = np.zeros((d, n))
    M_flat = M.flatten()
    M_flat[omega] = MM.flatten()[omega]
    M = M_flat.reshape(d, n)
    
    X = np.zeros((d, n))
    E = X.copy()
    Y = X.copy()

    nuclearnormX = 0

    for iteration in range(1, max_iter + 1):
        Xk = X.copy()
        Ek = E.copy()

        X, nuclearnormX = prox_nuclear(-(E - M + Y / mu), 1 / mu)

        E = -(X - M + Y / mu)
        E_flat = E.flatten()
        E_flat[omega] = 0
        E = E_flat.reshape(d, n)

        dY = X + E - M

        chgX = np.max(np.abs(Xk - X))
        chgE = np.max(np.abs(Ek - E))
        chg = np.max([chgX, chgE, np.max(np.abs(dY))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = nuclearnormX
                err = np.linalg.norm(dY, 'fro')
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y = Y + mu * dY
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

    obj = nuclearnormX
    err = np.linalg.norm(dY, 'fro')
    return X, obj, err, iteration
