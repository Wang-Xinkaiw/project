import numpy as np


def nmodeproduct(A, M, n):
    """
    Calculates the n-Mode Product of a Tensor A and a Matrix M

    B = nmodeproduct(A, M, n)

    B = A (x)_n M .. According to the Definition in De Lathauwer (2000)

    with:
    A:    (I_1 x I_2 x .. I_n x .. I_N) .. ->  n is in [1..N]
    M:    (J   x I_n)
    B:    (I_1 x I_2 x .. J x   .. I_N)

    note: "(x)_n" is the operator between the tensor and the matrix

    version 0.001 - 2009 by Fabian Schneiter
    """
    dimvec = np.array(A.shape)
    n = int(n)

    if len(dimvec) < n or n < 1:
        raise ValueError('nmodeproduct: n is not within the order range of tensor A')
    if M.shape[1] != dimvec[n - 1]:
        raise ValueError('nmodeproduct: dimension n of tensor A is not equal to dimension 2 of matrix M')

    Ash = np.moveaxis(A, n - 1, 0)

    dimvecB = list(Ash.shape)
    dimvecB[0] = M.shape[0]

    B = M @ Ash.reshape(Ash.shape[0], -1)

    B = B.reshape(dimvecB)

    # Move the first axis back to position n
    B = np.moveaxis(B, 0, n - 1)

    return B
