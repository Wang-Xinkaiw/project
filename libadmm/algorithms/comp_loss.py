import numpy as np


def comp_loss(E, loss):
    """
    Compute loss function value

    Parameters:
    -----------
    E : ndarray
        matrix
    loss : str
        loss type: 'l1', 'l21', or 'l2'

    Returns:
    --------
    out : float
        loss value
    """
    if loss == 'l1':
        out = np.sum(np.abs(E))
    elif loss == 'l21':
        out = 0
        for i in range(E.shape[1]):
            out = out + np.linalg.norm(E[:, i])
    elif loss == 'l2':
        out = 0.5 * np.linalg.norm(E, 'fro') ** 2
    return out
