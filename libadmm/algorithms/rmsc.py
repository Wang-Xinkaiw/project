import numpy as np
from ..proximal_operators import prox_l1, project_simplex,prox_nuclear


def rmsc(X, lambda_, opts=None, strategy=None):
    """
    Solve the Robust Multi-view Spectral Clustering (RMSC) problem by M-ADMM

    min_{L,S_i} ||L||_*+lambda*sum_i ||S_i||_1,
    s.t. X_i=L+S_i, i=1,...,m, L>=0, L1=1.

    version 1.0 - 19/06/2016

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

    d, n, m = X.shape

    L = np.zeros((d, n))
    S = np.zeros((d, n, m))
    Z = L.copy()
    Y = S.copy()
    dY = S.copy()
    Y2 = L.copy()

    nuclearnormZ = 0

    for iteration in range(1, max_iter + 1):
        Lk = L.copy()
        Sk = S.copy()
        Zk = Z.copy()

        Z, nuclearnormZ = prox_nuclear(L + Y2 / mu, 1 / mu)

        for i in range(m):
            S[:, :, i] = prox_l1(-L + X[:, :, i] - Y[:, :, i] / mu, lambda_ / mu)

        temp = (np.sum(X - S - Y / mu, axis=2) + Z - Y2 / mu) / (m + 1)
        L = project_simplex(temp)

        for i in range(m):
            dY[:, :, i] = L + S[:, :, i] - X[:, :, i]

        dY2 = L - Z

        chgL = np.max(np.abs(Lk - L))
        chgZ = np.max(np.abs(Zk - Z))
        chgS = np.max(np.abs(Sk - S))
        chg = np.max([chgL, chgS, chgZ, np.max(np.abs(dY)), np.max(np.abs(dY2))])

        if DEBUG:
            if iteration == 1 or iteration % 10 == 0:
                obj = nuclearnormZ + lambda_ * np.sum(np.abs(S))
                err = np.sqrt(np.linalg.norm(dY) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
                print(f'iter {iteration}, mu={mu}, obj={obj}, err={err}')

        if chg < tol:
            break

        Y = Y + mu * dY
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

                # 使用策略更新 mu（如果提供了 strategy 函数）
                if strategy is not None:
                    try:
                        iteration_state = {
                            'iteration': iteration,
                            'primal_residual': float(np.linalg.norm(dY1, 'fro')),
                            'dual_residual': float(np.linalg.norm(dY2, 'fro')),
                            'beta': mu,
                            'objective': float(obj) if 'obj' in locals() else 0.0,
                            'converged': False
                        }
                        mu = float(strategy(iteration_state))
                        mu = min(max(mu, 1e-10), max_mu)
                    except Exception:
                        # 如果策略调用失败，使用原有更新规则
                        mu = min(rho * mu, max_mu)
                else:
                    # 原有更新规则
                    mu = min(rho * mu, max_mu)

        else:

            # 使用策略更新 mu（如果提供了 strategy 函数）
            if strategy is not None:
                try:
                    iteration_state = {
                        'iteration': iteration,
                        'primal_residual': float(np.linalg.norm(dY1, 'fro')),
                        'dual_residual': float(np.linalg.norm(dY2, 'fro')),
                        'beta': mu,
                        'objective': float(obj) if 'obj' in locals() else 0.0,
                        'converged': False
                    }
                    mu = float(strategy(iteration_state))
                    mu = min(max(mu, 1e-10), max_mu)
                except Exception:
                    # 如果策略调用失败，使用原有更新规则
                    mu = min(rho * mu, max_mu)
            else:
                # 原有更新规则
                mu = min(rho * mu, max_mu)

    obj = nuclearnormZ + lambda_ * np.sum(np.abs(S))
    err = np.sqrt(np.linalg.norm(dY) ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return L, S, obj, err, iteration
