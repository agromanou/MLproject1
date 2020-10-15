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

def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

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


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """

    :param y:
    :param x:
    :param k_indices:
    :param k:
    :param lambda_:
    :param degree:
    :return:
    """
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    # ridge regression
    w = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
    return loss_tr, loss_te, w


if __name__ == '__main__':
    x = np.array([[1, 2], [3, 4], [3, 4], [5, 6], [5, 6],
                  [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    print(x)
    print(y)

    folds = 5
    k_indices = build_k_indices(y, folds)

    for k in range(folds):
        te_indice = k_indices[k]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indice = tr_indice.reshape(-1)
        print(te_indice)
        print(tr_indice)
