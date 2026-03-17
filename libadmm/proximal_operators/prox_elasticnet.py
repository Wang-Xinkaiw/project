import numpy as np


def prox_elasticnet(b, lambda1, lambda2):
    """
    The proximal operator of the elastic net

    min_x lambda1*||x||_1 + 0.5*lambda2*||x||_2^2 + 0.5*||x-b||_2^2

    version 1.0 - 18/06/2016

    Written by Canyi Lu (canyilu@gmail.com)
    """
    return (np.maximum(0, b - lambda1) + np.minimum(0, b + lambda1)) / (lambda2 + 1)
