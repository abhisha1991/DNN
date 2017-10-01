# GRADED FUNCTION: compute_cost
import numpy as np


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation:
    cost = −1m∑i=1m(y(i)log(a[L](i))+(1−y(i))log(1−a[L](i)))

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    A = np.dot(Y, np.log(AL.T))  # Y = (1, m), AL = (1, m) => A = (1, 1)
    # print(A.shape)
    B = np.dot((1 - Y), np.log(1 - AL.T))  # Y = (1, m), AL = (1, m) => B = (1, 1)
    # print(B.shape)
    A = float(np.squeeze(A))
    B = float(np.squeeze(B))
    cost = -1 / m * (A + B)

    cost = np.squeeze(cost)  # To make the cost shape into a single number (e.g. this turns [[17]] into 17).
    cost = np.asarray(cost)
    assert (cost.shape == ())

    return cost