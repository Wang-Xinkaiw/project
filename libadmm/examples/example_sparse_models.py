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

from libadmm.algorithms import l1, groupl1, elasticnet, fusedl1, tracelasso, ksupport
from libadmm.algorithms import l1R, groupl1R, elasticnetR, fusedl1R, tracelassoR, ksupportR


def example_sparse_models():
    """
    Examples for testing the sparse models
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
        'rho': 1.1,
        'mu': 1e-4,
        'max_mu': 1e10,
        'DEBUG': 0
    }

    print("=" * 60)
    print("Example 1: l1")
    print("=" * 60)
    X2, obj, err, iter_count = l1(A, B, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 2: group l1")
    print("=" * 60)
    g_num = 5
    g_len = int(na / g_num)
    G = []
    for i in range(g_num - 1):
        G.append(np.arange(i * g_len, (i + 1) * g_len))
    G.append(np.arange((g_num - 1) * g_len, na))

    X2, obj, err, iter_count = groupl1(A, B, G, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 3: elastic net")
    print("=" * 60)
    lambda_ = 0.01
    X2, obj, err, iter_count = elasticnet(A, B, lambda_, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 4: fused Lasso")
    print("=" * 60)
    lambda_ = 0.01
    x, obj, err, iter_count = fusedl1(A, b, lambda_, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 5: trace Lasso")
    print("=" * 60)
    x, obj, err, iter_count = tracelasso(A, b, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 6: k-support norm")
    print("=" * 60)
    k = 10
    X, err, iter_count = ksupport(A, B, k, opts)
    print(f"iter: {iter_count}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 7: regularized l1")
    print("=" * 60)
    lambda_ = 0.01
    opts['loss'] = 'l1'
    X, E, obj, err, iter_count = l1R(A, B, lambda_, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 8: regularized group Lasso")
    print("=" * 60)
    g_num = 5
    g_len = int(na / g_num)
    G = []
    for i in range(g_num - 1):
        G.append(np.arange(i * g_len, (i + 1) * g_len))
    G.append(np.arange((g_num - 1) * g_len, na))
    lambda_ = 1.0
    opts['loss'] = 'l1'
    X, E, obj, err, iter_count = groupl1R(A, B, G, lambda_, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 9: regularized elastic net")
    print("=" * 60)
    lambda1 = 10.0
    lambda2 = 10.0
    opts['loss'] = 'l1'
    X, E, obj, err, iter_count = elasticnetR(A, B, lambda1, lambda2, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 10: regularized fused Lasso")
    print("=" * 60)
    lambda1 = 10.0
    lambda2 = 10.0
    opts['loss'] = 'l1'
    X, E, obj, err, iter_count = fusedl1R(A, b, lambda1, lambda2, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 11: regularized trace Lasso")
    print("=" * 60)
    lambda_ = 0.1
    opts['loss'] = 'l1'
    x, e, obj, err, iter_count = tracelassoR(A, b, lambda_, opts)
    print(f"iter: {iter_count}")
    print(f"obj: {obj}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("Example 12: regularized k-support norm")
    print("=" * 60)
    lambda_ = 0.1
    k = 10
    X, E, err, iter_count = ksupportR(A, B, lambda_, k, opts)
    print(f"iter: {iter_count}")
    print(f"err: {err}")

    print("\n" + "=" * 60)
    print("All sparse model examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    example_sparse_models()
