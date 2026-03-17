"""
References:

C. Lu. A Library of ADMM for Sparse and Low-rank Optimization. National University of Singapore, June 2016.
https://github.com/canyilu/LibADMM.
C. Lu, J. Feng, S. Yan, Z. Lin. A Unified Alternating Direction Method of Multipliers by Majorization
Minimization. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 527-541, 2018
"""

import numpy as np
import sys
sys.path.insert(0, 'c:/Users/R/Desktop/一个文件夹/project')

from libadmm.algorithms import rpca, lrmc, lrmcR, lrr, latlrr, lrsr, igc, rmsc, sparsesc


def example_low_rank_matrix_models():
    """
    Examples for testing the low-rank matrix based models
    For detailed description of the sparse models, please refer to the Manual.
    """

    np.random.seed(42)

    d = 10
    na = 200
    nb = 100

    A = np.random.randn(d, na)
    X = np.random.randn(na, nb)
    B = A @ X
    b = B[:, 0]

    opts = {
        'tol': 1e-6,
        'max_iter': 1000,
        'rho': 1.2,
        'mu': 1e-3,
        'max_mu': 1e10,
        'DEBUG': 0
    }

    print("=" * 60)
    print("Example 1: RPCA")
    print("=" * 60)
    n1 = 100
    n2 = 200
    r = 10
    L = np.random.randn(n1, r) @ np.random.randn(r, n2)

    p = 0.1
    m = int(p * n1 * n2)
    temp = np.random.rand(n1 * n2)
    I = np.argsort(temp)
    I = I[:m]
    Omega = np.zeros((n1, n2))
    Omega_flat = Omega.flatten()
    Omega_flat[I] = 1
    Omega = Omega_flat.reshape(n1, n2)
    E = np.sign(np.random.rand(n1, n2) - 0.5)
    S = Omega * E

    Xn = L + S

    lambda_ = 1 / np.sqrt(max(n1, n2))
    opts['loss'] = 'l1'
    opts['DEBUG'] = 1

    Lhat, Shat, obj, err, iter_count = rpca(Xn, lambda_, opts)
    rel_err_L = np.linalg.norm(L - Lhat, 'fro') / np.linalg.norm(L, 'fro')
    rel_err_S = np.linalg.norm(S - Shat, 'fro') / np.linalg.norm(S, 'fro')
    print(f"rel_err_L: {rel_err_L}")
    print(f"rel_err_S: {rel_err_S}")
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 2: Low-rank matrix completion (lrmc)")
    print("=" * 60)
    n1 = 100
    n2 = 200
    r = 5
    X = np.random.randn(n1, r) @ np.random.randn(r, n2)

    p = 0.6
    omega = np.where(np.random.rand(n1, n2) < p)
    M = np.zeros((n1, n2))
    M[omega] = X[omega]

    Xhat, obj, err, iter_count = lrmc(M, omega, opts)
    rel_err_X = np.linalg.norm(Xhat - X, 'fro') / np.linalg.norm(X, 'fro')
    print(f"rel_err_X: {rel_err_X}")

    print("\n" + "=" * 60)
    print("Example 3: Regularized lrmc")
    print("=" * 60)
    E = np.random.randn(n1, n2) / 100
    M = X + E
    lambda_ = 0.1
    Xhat, E,obj, err, iter_count = lrmcR(M, omega, lambda_, opts)
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 4: Low-rank representation (lrr)")
    print("=" * 60)
    lambda_ = 0.001
    opts['loss'] = 'l21'
    X, E, obj, err, iter_count = lrr(A, A, lambda_, opts)
    print(f"obj: {obj}")
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 5: Latent LRR (latlrr)")
    print("=" * 60)
    lambda_ = 0.1
    opts['loss'] = 'l1'
    Z, L, obj, err, iter_count = latlrr(A, lambda_, opts)
    print(f"obj: {obj}")
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 6: Low-rank and sparse representation (lrsr)")
    print("=" * 60)
    lambda1 = 0.1
    lambda2 = 4.0
    opts['loss'] = 'l21'
    X, E, obj, err, iter_count = lrsr(A, B, lambda1, lambda2, opts)
    print(f"obj: {obj}")
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 7: Improved graph clustering (igc)")
    print("=" * 60)
    n = 100
    r = 5
    X = np.random.randn(n, r) @ np.random.randn(r, n)
    C = np.abs(np.random.randn(n, n))
    lambda_ = 1 / np.sqrt(n)
    opts['loss'] = 'l1'
    opts['DEBUG'] = 1
    L, S, obj, err, iter_count = igc(X, C, lambda_, opts)
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 8: Robust multi-view spectral clustering (rmsc)")
    print("=" * 60)
    n = 100
    r = 5
    m = 10
    X = np.random.randn(n, n, m)
    lambda_ = 1 / np.sqrt(n)
    opts['loss'] = 'l1'
    opts['DEBUG'] = 1
    L, S, obj, err, iter_count = rmsc(X, lambda_, opts)
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 9: Sparse spectral clustering (sparsesc)")
    print("=" * 60)
    lambda_ = 0.001
    n = 100
    X = np.random.randn(n, n)
    W = np.abs(X.T @ X)
    I = np.eye(n)
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    L = I - D_inv_sqrt @ W @ D_inv_sqrt
    k = 5
    P, obj, err, iter_count = sparsesc(L, lambda_, k, opts)
    print(f"obj: {obj}")
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("All low-rank matrix model examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    example_low_rank_matrix_models()
