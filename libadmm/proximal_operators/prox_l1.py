import numpy as np


def prox_l1(b, lambda_):
    """
    The proximal operator of the l1 norm

    min_x lambda*||x||_1 + 0.5*||x-b||_2^2

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)
    """
    return np.maximum(0, b - lambda_) + np.minimum(0, b + lambda_)
