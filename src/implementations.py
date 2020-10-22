import numpy as np
from .implementation_helpers import *

# The six compulsory learning methods

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Least squares regression using gradient descent
    Args:
        y: labels
        tx: features
        initial_w: initial weights
        max_iters: maximum number of iterations
        gamma: step size
    Returns:
        w: optimal weight
        loss: optimal loss
    """

    weights = [initial_w]
    losses = []
    w = initial_w
    thres = 1e-8

    for i in range(max_iters):
        error_vector = get_error_vector(y, tx, w)
        gradient_vector = get_gradient(tx, error_vector)
        loss = get_mse(error_vector)
        w = w - gamma * gradient_vector
        weights.append(w)
        losses.append(loss)
        if len(losses) > 1:
            if np.abs(losses[-1] - losses[-2]) < thres:
                break
    return weights[-1], losses[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Least squares regression using stochastic gradient descent
    Args:
        y: labels
        tx: features
        initial_w: initial weights
        max_iters: maximum number of iterations
        gamma: step size
    Returns:
        w: optimal weight
        loss: optimal loss
    """

    weights = [initial_w]
    losses = []
    w = initial_w
    thres = 1e-8

    for i in range(max_iters):
        random_index = np.random.randint(len(y))

        y_random = y[random_index]

        tx_random = tx[random_index]

        error_vector = get_error_vector(y_random, tx_random, w)
        gradient_vector = get_gradient(tx_random, error_vector)
        loss = get_mse(error_vector)

        w = w - gamma * gradient_vector
        weights.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < thres:
            break # convergence criterion met
    return weights[-1], losses[-1]

def least_squares(y, tx):
    """
    Least squares regression
    Args:
        y: labels
        tx: features
    Returns:
        w: optimal weight
        loss: optimal loss
    """
    c_m = tx.T.dot(tx)
    c_v = tx.T.dot(y)

    w = np.linalg.solve(c_m, c_v)
    error_vector = get_error_vector(y, tx, w)
    loss = get_mse(error_vector)

    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Args:
        y: labels
        tx: features
        lambda_: regularization hyperparameter
    Returns:
        w: optimal weight
        loss: optimal loss
    """
    c_m = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    c_v = tx.T.dot(y)

    w = np.linalg.solve(c_m, c_v)
    error_vector = get_error_vector(y, tx, w)
    loss = get_mse(error_vector)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using stochastic gradient descent
    Args:
        y: labels
        tx: features
        initial_w: initial weights vector
        max_iters: maximum number of iterations
        gamma: step size
    Returns:
        w: optimal weight
        loss: optimal loss
    """
    weights = [initial_w]
    thres = 1e-8
    losses = []
    w = initial_w
    for i in range(max_iters):
        loss = get_logistic_loss(y, tx, w)
        gradient_vector = get_logistic_gradient(y, tx, w)
        w = w - gamma * gradient_vector
        weights.append(w)
        losses.append(loss)
        if len(losses) > 1:
            if np.abs(losses[-1] - losses[-2]) < thres:
                break
    return weights[-1], losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent
    Args:
        y: labels
        tx: features
        lambda_: regularization hyperparameter
        initial_w: initial weights
        max_iters: maximum number of iterations
        gamma: step size
    Returns:
        w: optimal weight
        loss: optimal loss
    """

    weights = [initial_w]
    losses = []
    w = initial_w
    thres = 1e-8
    for i in range(max_iters):
        loss, gradient_vector = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient_vector
        weights.append(w)
        losses.append(loss)

        if len(losses) > 1:
            if np.abs(losses[-1] - losses[-2]) < thres:
                break
    return weights[-1], losses[-1]
