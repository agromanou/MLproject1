from costs import *
from proj1_helpers import batch_iter


# The six compulsory learning methods

def least_squares_GD(y, tx, initial_w, max_iters, gamma, verbose=False):
    """
    Linear regression using gradient descent.

    :param y: np.array with the labels
    :param tx: np.array with the features
    :param initial_w: np.array with the initial weights
    :param max_iters: int, maximum number of iterations
    :param gamma: float, step size
    :param verbose: boolean, prints losses every 100 iterations
    :returns:
        w: np.array with the optimal weights
        loss: float, optimal loss
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    threshold = 1e-8

    for i in range(max_iters):
        # Compute loss
        err = calculate_error(y, tx, w)
        loss = calculate_mse(err)

        # Compute the gradient for mse loss
        gradient_vector = calculate_gradient(tx, err)

        # Update weights
        w -= gamma * gradient_vector

        ws.append(w)
        losses.append(loss)

        if i % 100 == 0:
            print("Current iteration of GD={i}, loss={loss:.4f}".format(i=i, loss=loss)) if verbose else None

        # convergence criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return ws[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, verbose=False):
    """
    Linear regression using stochastic gradient descent.

    :param y: np.array with the labels
    :param tx: np.array with the features
    :param initial_w: np.array with the initial weights
    :param max_iters: int, maximum number of iterations
    :param gamma: float, step size
    :param verbose: boolean, prints losses every 100 iterations
    :returns:
        w: np.array with the optimal weights
        loss: float, optimal loss
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    threshold = 1e-8

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # Compute loss
            err = calculate_error(y_batch, tx_batch, w)
            loss = calculate_mse(err)

            # Compute the gradient for mse loss
            gradient_vector = calculate_gradient(tx_batch, err)

            # Update weights
            w -= gamma * gradient_vector

            ws.append(w)
            losses.append(loss)

            if i % 100 == 0:
                print("Current iteration of SGD={i}, loss={loss:.4f}".format(i=i, loss=loss)) if verbose else None

            # convergence criterion
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
    return ws[-1], losses[-1]


def least_squares(y, tx):
    """
    Least squares regression using normal equations.

    :param y: np.array with the labels
    :param tx: np.array with the features
    :return:
        w: np.array with the optimal weights
        loss: float, optimal loss
    """
    c_m = tx.T.dot(tx)
    c_v = tx.T.dot(y)

    # Calculate the least squares
    w = np.linalg.solve(c_m, c_v)

    # Compute loss
    err = calculate_error(y, tx, w)
    loss = calculate_mse(err)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    :param y: np.array with the labels
    :param tx: np.array with the features
    :param lambda_: float, regularization hyper-parameter
    :return:
        w: np.array with the optimal weights
        loss: float, optimal loss
    """
    c_m = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    c_v = tx.T.dot(y)

    # Calculate the least squares
    w = np.linalg.solve(c_m, c_v)

    # Compute loss
    err = calculate_error(y, tx, w)
    loss = calculate_mse(err)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, batch_size=0, verbose=False):
    """
    Logistic regression using stochastic gradient descent or gradient descent.

    :param y: np.array with the labels
    :param tx: np.array with the features
    :param initial_w: np.array with the initial weights
    :param max_iters: int, maximum number of iterations
    :param gamma: float, step size
    :param batch_size: int, if 0 it runs GD
    :param verbose: boolean, prints losses every 100 iterations
    :returns:
        w: np.array with the optimal weights
        loss: float, optimal loss
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    threshold = 1e-8

    # if no batch size is given it runs GD
    if not batch_size:
        batch_size = len(tx)

    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # Compute loss
            loss = calculate_logistic_loss(y_batch, tx_batch, w)

            # Compute the gradient for mse loss
            gradient_vector = calculate_logistic_gradient(y, tx, w)

            # Update weights
            w -= gamma * gradient_vector

            ws.append(w)
            losses.append(loss)

            if i % 100 == 0:
                print("Current iteration of SGD={i}, loss={loss:.4f}".format(i=i, loss=loss)) if verbose else None

            if len(losses) > 1:
                if np.abs(losses[-1] - losses[-2]) < threshold:
                    break

    return ws[-1], losses[-1]




#######

def sigmoid(t):
    return 1/(1+np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_reg_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    #init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.

        loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = calculate_reg_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w.squeeze(), loss


####

# def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, batch_size=0, verbose=False):
#     """
#     Regularized logistic regression using gradient descent or SGD
#
#     :param y: np.array with the labels
#     :param tx: np.array with the features
#     :param lambda_: float, regularization hyper-parameter
#     :param initial_w: np.array with the initial weights
#     :param max_iters: int, maximum number of iterations
#     :param gamma: float, step size
#     :param batch_size: int, if 0 it runs GD
#     :param verbose: boolean, prints losses every 100 iterations
#     :returns:
#         w: np.array with the optimal weights
#         loss: float, optimal loss
#     """
#     ws = [initial_w]
#     losses = []
#     w = initial_w
#     threshold = 1e-8
#
#     # if no batch size is given it runs GD
#     if not batch_size:
#         batch_size = len(tx)
#
#     for i in range(max_iters):
#         for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
#             # Compute the loss and the gradient for mse loss
#             loss, gradient_vector = penalized_logistic_regression(y_batch, tx_batch, w, lambda_)
#
#             # Update weights
#             w -= gamma * gradient_vector
#
#             ws.append(w)
#             losses.append(loss)
#
#             if i % 100 == 0:
#                 print("Current iteration={i}, loss={loss:.4f}" .format(i=i, loss=loss)) if verbose else None
#
#             if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
#                 break
#
#     return ws[-1], losses[-1]
