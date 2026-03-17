import numpy as np


def _bsearch(z, array, low, high, diff, k, r, L):
    """Helper function for k support norm proximal operator"""
    if z[low] == 0:
        l = low
        T = 0
        return l, T

    while low < high:
        mid = int(np.floor((low + high) / 2)) + 1
        tmp = mid - k + r + 1 + L * (r + 1)
        if z[mid] * tmp - (array[mid] - diff) > 0:
            low = mid
        else:
            high = mid - 1
    l = low
    T = array[low] - diff
    return l, T


def prox_ksupport(v, k, lambda_):
    """
    The proximal operator of the k support norm of a vector

    min_x 0.5*lambda*||x||_{ksp}^2 + 0.5*||x-v||_2^2

    version 1.0 - 27/06/2016

    Written by Hanjiang Lai

    Reference:
    Lai H, Pan Y, Lu C, et al. Efficient k-support matrix pursuit, ECCV, 2014: 617-631.
    """
    L = 1 / lambda_
    d = len(v)

    if k >= d:
        return L * v / (1 + L)

    if k <= 1:
        k = 1

    z = np.sort(np.abs(v))[::-1].copy()
    z = z * L
    ar = np.cumsum(z)
    z = np.append(z, -np.inf)

    diff = 0
    err = np.inf
    found = False

    for r in range(k - 1, -1, -1):
        l, T = _bsearch(z, ar, k - r - 1 if k - r - 1 >= 0 else 0, d - 1, diff, k, r, L)
        if l > d - 1:
            l = d - 1
        if l >= 0 and l < len(z) and k - r - 1 >= 0 and k - r - 1 < len(z):
            if ((L + 1) * T >= (l - k + (L + 1) * r + L + 1) * z[k - r] and
                ((k - r - 1 == 0) or (L + 1) * T < (l - k + (L + 1) * r + L + 1) * z[k - r - 1])):
                found = True
                break
        diff = diff + z[k - r] if k - r < len(z) else diff
        if k - r - 1 == 0:
            err_tmp = max(0, (l - k + (L + 1) * r + L + 1) * z[k - r] - (L + 1) * T) if k - r < len(z) and l >= 0 else 0
        else:
            err_tmp = 0
            if k - r < len(z) and l >= 0:
                err_tmp += max(0, (l - k + (L + 1) * r + L + 1) * z[k - r] - (L + 1) * T)
            if k - r - 1 < len(z):
                err_tmp += max(0, -(l - k + (L + 1) * r + L + 1) * z[k - r - 1] + (L + 1) * T)
        if err > err_tmp:
            err_r = r
            err_l = l
            err_T = T
            err = err_tmp

    if not found:
        r = err_r
        l = err_l
        T = err_T

    p = np.zeros(d)
    if k - r - 1 > 0:
        p[:min(k - r - 1, d)] = z[:min(k - r - 1, d)] / (L + 1)
    if l >= k - r and l < d:
        if l - (k - r - 1) > 0 and l < d:
            end_idx = min(l + 1, d)
            start_idx = k - r - 1
            if start_idx < end_idx:
                p[start_idx:end_idx] = T / (l - k + (L + 1) * r + L + 1)
    if l + 1 < d:
        p[l + 1:d] = z[l + 1:d]

    ind = np.argsort(np.abs(v))[::-1]
    rev = np.zeros(d, dtype=int)
    rev[ind] = np.arange(d)
    p = p[rev]

    B = v - (1 / L) * p * np.sign(v)
    return B
