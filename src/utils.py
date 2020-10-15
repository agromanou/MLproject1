'''
useful functions for pipeline
'''

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def build_k_indices(y, folds, seed=1):
    """
    Builds k indices for k-fold cross validation.
    If the number of folds are larger than the length of y,
    the function returns an empty list.

    :param y: a numpy array, representing the given labels
    :param folds: int, the value of k in k-fold validations, must be greater than zero
    :param seed: int, seed number for the shuffling
    :return:
    """
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_rows = len(y)
    # find fold intervals
    interval = int(num_rows / folds)
    # shuffle indexes and split them based on the intervals
    indices = np.random.permutation(num_rows)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(folds)]
    return np.array(k_indices)

