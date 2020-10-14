'''
useful functions for pipeline
'''

import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    # compute loss for each combination of w0 and w1.
    for ind_row, row in enumerate(w0):
        for ind_col, col in enumerate(w1):
            w = np.array([row, col])
            losses[ind_row, ind_col] = compute_loss(y, tx, w)
    return losses


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







