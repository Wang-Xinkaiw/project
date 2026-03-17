import numpy as np


def tprod(A, B):
    """
    Tensor-tensor product of two 3 way tensors: C = A*B
    A - n1*n2*n3 tensor
    B - n2*l*n3  tensor
    C - n1*l*n3  tensor

    version 2.0 - 09/10/2017

    Written by Canyi Lu (canyilu@gmail.com)

    References:
    Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University.
    June, 2018. https://github.com/canyilu/tproduct.

    Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
    Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
    Norm, arXiv preprint arXiv:1804.03728, 2018
    """
    n1, n2, n3 = A.shape
    m1, m2, m3 = B.shape

    if n2 != m1 or n3 != m3:
        raise ValueError('Inner tensor dimensions must agree.')

    A_fft = np.fft.fft(A, axis=2)
    B_fft = np.fft.fft(B, axis=2)
    C = np.zeros((n1, m2, n3), dtype=A.dtype)

    C_fft = np.zeros((n1, m2, n3), dtype=A_fft.dtype)
    C_fft[:, :, 0] = A_fft[:, :, 0] @ B_fft[:, :, 0]

    halfn3 = int(np.floor(n3 / 2))
    for i in range(1, halfn3 + 1):
        C_fft[:, :, i] = A_fft[:, :, i] @ B_fft[:, :, i]
        if i < n3 - i:
            C_fft[:, :, n3 - i] = np.conj(C_fft[:, :, i])

    if n3 % 2 == 1:
        i = halfn3
        if i < n3:
            C_fft[:, :, i] = A_fft[:, :, i] @ B_fft[:, :, i]

    C = np.fft.ifft(C_fft, axis=2)
    if np.iscomplexobj(C):
        C = np.real(C)
    return C
