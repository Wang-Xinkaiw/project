import numpy as np


def tran(X):
    """
    conjugate transpose of a 3 way tensor
    X  - n1*n2*n3 tensor
    Xt - n2*n1*n3  tensor

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)

    References:
    Canyi Lu, Tensor-Tensor Product Toolbox. Carnegie Mellon University.
    June, 2018. https://github.com/canyilu/tproduct.

    Canyi Lu, Jiashi Feng, Yudong Chen, Wei Liu, Zhouchen Lin and Shuicheng
    Yan, Tensor Robust Principal Component Analysis with A New Tensor Nuclear
    Norm, arXiv preprint arXiv:1804.03728, 2018
    """
    n1, n2, n3 = X.shape
    Xt = np.zeros((n2, n1, n3), dtype=X.dtype)
    Xt[:, :, 0] = X[:, :, 0].T
    for i in range(1, n3):
        Xt[:, :, i] = X[:, :, n3 - i].T
    return Xt
