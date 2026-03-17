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

from libadmm.algorithms import trpca_snn, lrtc_snn, lrtcR_snn, trpca_tnn, lrtc_tnn, lrtcR_tnn, lrtr_Gaussian_tnn
from libadmm.tensor_tools import nmodeproduct, tprod, tubalrank


def example_low_rank_tensor_models():
    """
    Examples for testing the low-rank tensor models
    For detailed description of the sparse models, please refer to the Manual.
    """

    np.random.seed(42)

    # Set different mu values for different models
    snn_opts = {
        'mu': 1e-4,
        'rho': 1.1,
        'max_iter': 500,
        'DEBUG': 1
    }
    tnn_opts = {
        'mu': 1e-4,
        'rho': 1.1,
        'max_iter': 500,
        'DEBUG': 1
    }

    print("=" * 60)
    print("Example 1: Tensor RPCA based on SNN (trpca_snn)")
    print("=" * 60)
    n1 = 50
    n2 = n1
    n3 = n1
    r = 5
    L = np.random.rand(r, r, r)
    U1 = np.random.rand(n1, r)
    U2 = np.random.rand(n2, r)
    U3 = np.random.rand(n3, r)
    L = nmodeproduct(L, U1, 1)
    L = nmodeproduct(L, U2, 2)
    L = nmodeproduct(L, U3, 3)

    p = 0.05
    m = int(p * n1 * n2 * n3)
    temp = np.random.rand(n1 * n2 * n3)
    I = np.argsort(temp)
    I = I[:m]
    Omega = np.zeros((n1, n2, n3))
    Omega_flat = Omega.flatten()
    Omega_flat[I] = 1
    Omega = Omega_flat.reshape(n1, n2, n3)
    E = np.sign(np.random.rand(n1, n2, n3) - 0.5)
    S = Omega * E

    Xn = L + S

    lambda_ = np.array([1.0, 1.0, 1.0])
    Lhat, Shat, err, iter_count = trpca_snn(Xn, lambda_, snn_opts)
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 2: Low-rank tensor completion based on SNN (lrtc_snn)")
    print("=" * 60)
    n1 = 50
    n2 = n1
    n3 = n1
    r = 5
    X = np.random.rand(r, r, r)
    U1 = np.random.rand(n1, r)
    U2 = np.random.rand(n2, r)
    U3 = np.random.rand(n3, r)
    X = nmodeproduct(X, U1, 1)
    X = nmodeproduct(X, U2, 2)
    X = nmodeproduct(X, U3, 3)
    p = 0.5
    omega = np.where(np.random.rand(n1 * n2 * n3) < p)[0]
    M = np.zeros((n1, n2, n3))
    M_flat = M.flatten()
    X_flat = X.flatten()
    M_flat[omega] = X_flat[omega]
    M = M_flat.reshape(n1, n2, n3)

    lambda_ = np.array([1.0, 1.0, 1.0])
    Xhat, err, iter_count = lrtc_snn(M, omega, lambda_, snn_opts)
    print(f"err: {err}")
    print(f"iter: {iter_count}")
    RSE = np.linalg.norm(X.flatten() - Xhat.flatten()) / np.linalg.norm(X.flatten())
    print(f"RSE: {RSE}")

    print("\n" + "=" * 60)
    print("Example 3: Regularized LRTC based on SNN (lrtcR_snn)")
    print("=" * 60)
    n1 = 50
    n2 = n1
    n3 = n1
    r = 5
    X = np.random.rand(r, r, r)
    U1 = np.random.rand(n1, r)
    U2 = np.random.rand(n2, r)
    U3 = np.random.rand(n3, r)
    X = nmodeproduct(X, U1, 1)
    X = nmodeproduct(X, U2, 2)
    X = nmodeproduct(X, U3, 3)
    p = 0.5
    omega = np.where(np.random.rand(n1 * n2 * n3) < p)[0]
    M = np.zeros((n1, n2, n3))
    M_flat = M.flatten()
    X_flat = X.flatten()
    M_flat[omega] = X_flat[omega]
    M = M_flat.reshape(n1, n2, n3)

    lambda_ = np.array([1.0, 1.0, 1.0])
    Xhat, err, iter_count = lrtcR_snn(M, omega, lambda_, snn_opts)
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 4: Tensor RPCA based on TNN (trpca_tnn)")
    print("=" * 60)
    n1 = 50
    n2 = n1
    n3 = n1
    r = int(0.1 * n1)
    L1 = np.random.randn(n1, r, n3) / n1
    L2 = np.random.randn(r, n2, n3) / n2
    L = tprod(L1, L2)

    p = 0.1
    m = int(p * n1 * n2 * n3)
    temp = np.random.rand(n1 * n2 * n3)
    I = np.argsort(temp)
    I = I[:m]
    Omega = np.zeros((n1, n2, n3))
    Omega_flat = Omega.flatten()
    Omega_flat[I] = 1
    Omega = Omega_flat.reshape(n1, n2, n3)
    E = np.sign(np.random.rand(n1, n2, n3) - 0.5)
    S = Omega * E

    Xn = L + S
    lambda_ = 1 / np.sqrt(n3 * max(n1, n2))

    Lhat, Shat, obj, err, iter_count = trpca_tnn(Xn, lambda_, tnn_opts)
    RES_L = np.linalg.norm(L.flatten() - Lhat.flatten()) / np.linalg.norm(L.flatten())
    RES_S = np.linalg.norm(S.flatten() - Shat.flatten()) / np.linalg.norm(S.flatten())
    trank = tubalrank(Lhat)
    print(f"RES_L: {RES_L}")
    print(f"RES_S: {RES_S}")
    print(f"trank: {trank}")

    print("\n" + "=" * 60)
    print("Example 5: Low-rank tensor completion based on TNN (lrtc_tnn)")
    print("=" * 60)
    n1 = 50
    n2 = n1
    n3 = n1
    r = int(0.1 * n1)
    L1 = np.random.randn(n1, r, n3) / n1
    L2 = np.random.randn(r, n2, n3) / n2
    X = tprod(L1, L2)
    p = 0.5
    omega = np.where(np.random.rand(n1 * n2 * n3) < p)[0]
    M = np.zeros((n1, n2, n3))
    M_flat = M.flatten()
    X_flat = X.flatten()
    M_flat[omega] = X_flat[omega]
    M = M_flat.reshape(n1, n2, n3)

    Xhat, obj, err, iter_count = lrtc_tnn(M, omega, tnn_opts)
    print(f"err: {err}")
    print(f"iter: {iter_count}")
    RSE = np.linalg.norm(X.flatten() - Xhat.flatten()) / np.linalg.norm(X.flatten())
    trank = tubalrank(Xhat)
    print(f"RSE: {RSE}")
    print(f"trank: {trank}")

    print("\n" + "=" * 60)
    print("Example 6: Regularized LRTC based on TNN (lrtcR_tnn)")
    print("=" * 60)
    n1 = 50
    n2 = n1
    n3 = n1
    r = int(0.1 * n1)
    L1 = np.random.randn(n1, r, n3) / n1
    L2 = np.random.randn(r, n2, n3) / n2
    X = tprod(L1, L2)
    p = 0.5
    omega = np.where(np.random.rand(n1 * n2 * n3) < p)[0]
    M = np.zeros((n1, n2, n3))
    M_flat = M.flatten()
    X_flat = X.flatten()
    M_flat[omega] = X_flat[omega]
    M = M_flat.reshape(n1, n2, n3)

    lambda_ = 0.5
    Xhat, Ehat, obj, err, iter_count = lrtcR_tnn(M, omega, lambda_, tnn_opts)
    print(f"err: {err}")
    print(f"iter: {iter_count}")

    print("\n" + "=" * 60)
    print("Example 7: Low-rank tensor recovery from Gaussian measurements (lrtr_Gaussian_tnn)")
    print("=" * 60)
    n1 = 30
    n2 = n1
    n3 = 5
    r = int(0.2 * n1)
    X = tprod(np.random.randn(n1, r, n3), np.random.randn(r, n2, n3))

    m = int(3 * r * (n1 + n2 - r) * n3 + 1)
    n = n1 * n2 * n3
    A = np.random.randn(m, n) / np.sqrt(m)

    b = A @ X.flatten()
    Xsize = {'n1': n1, 'n2': n2, 'n3': n3}

    tnn_opts['DEBUG'] = 1
    Xhat, obj, err, iter_count = lrtr_Gaussian_tnn(A, b, Xsize, tnn_opts)
    RSE = np.linalg.norm(Xhat.flatten() - X.flatten()) / np.linalg.norm(X.flatten())
    trank = tubalrank(Xhat)
    print(f"RSE: {RSE}")
    print(f"trank: {trank}")

    print("\n" + "=" * 60)
    print("All low-rank tensor model examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    example_low_rank_tensor_models()
