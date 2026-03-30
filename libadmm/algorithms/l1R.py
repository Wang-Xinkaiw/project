import numpy as np
from ..proximal_operators import prox_l1
from .comp_loss import comp_loss


def l1R(A, B, lambda_, opts=None, strategy=None):
    """
    Solve the l1 norm regularized minimization problem by M-ADMM

    min_{X,E} loss(E)+lambda*||X||_1, s.t. AX+E=B
    loss(E) = ||E||_1 or 0.5*||E||_F^2

    Parameters:
    -----------
    A : ndarray
        d*na matrix
    B : ndarray
        d*nb matrix
    lambda_ : float
        >=0, parameter
    opts : dict, optional
        Structure value in Python. The fields are:
            opts.loss       -   'l1' (default): loss(E) = ||E||_1
                                'l2': loss(E) = 0.5*||E||_F^2
            opts.tol        -   termination tolerance
            opts.max_iter   -   maximum number of iterations
            opts.mu         -   stepsize for dual variable updating in ADMM
            opts.max_mu     -   maximum stepsize
            opts.rho        -   rho>=1, ratio used to increase mu
            opts.DEBUG      -   0 or 1
    strategy : object, optional
        Strategy object with update_parameters method for adaptive beta tuning

    strategy : callable, optional
        策略函数，签名为 adjust_beta(iteration_state: Dict[str, Any]) -> float
        用于在每次迭代中动态调整惩罚参数 mu

    Returns:
    --------
    X : ndarray
        na*nb matrix
    E : ndarray
        d*nb matrix
    obj : float
        objective function value
    err : float
        residual
    iter : int
        number of iterations

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

        X = prox_l1(Z - Y2 / mu, lambda_ / mu)

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
                obj = comp_loss(E, loss) + lambda_ * np.sum(np.abs(X))
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
                    'objective': float(comp_loss(E, loss) + lambda_ * np.sum(np.abs(X))),
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

    obj = comp_loss(E, loss) + lambda_ * np.sum(np.abs(X))
    err = np.sqrt(np.linalg.norm(dY1, 'fro') ** 2 + np.linalg.norm(dY2, 'fro') ** 2)
    return X, E, obj, err, iteration
