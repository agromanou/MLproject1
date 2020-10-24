import numpy as np


def calculate_error(y, tx, w):
    """
    Computes the error vector.

    :param y: np.array with the labels
    :param tx: np.array with the features
    :param w: np.array with the weights
    :returns:
        error_vector: np.array of the error vector
    """
    return y - tx.dot(w)


def calculate_mse(e):
    """
    Computes the mse for a given error vector.

    :param e: np.array of the error vector
    :returns:
        mse: float, mean squared error
    """
    return 1 / 2 * np.mean(e ** 2)


def calculate_gradient(tx, err):
    """
    Computes the gradient for a given error vector.

    :param tx: np.array with the features
    :param err: np.array of the error vector
    Returns:
        gradient: np.array of the gradient vector
    """
    return - tx.T.dot(err) / float(len(err))


def sigmoid(t):
    """
    Applies the sigmoid function to the input.

    :param t: np.array with the input
    :returns:
        sigmoid_t: np.array of sig(t)
    """
    return 1.0 / (1 + np.exp(-t))


def calculate_logistic_loss(y, tx, w):
    """
    Computes the logistic loss.

    :param y: np.array with the labels
    :param tx: np.array with the features
    :param w: np.array with the weights
    :returns:
        loss: float, logistic loss
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)


def calculate_logistic_gradient(y, tx, w):
    """
    Computes the gradient of the logistic regression loss function.

    :param y: np.array with the labels
    :param tx: np.array with the features
    :param w: np.array with the weights
    :returns:
        logistic_gradient: np.array of the gradient vector
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y)


def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Performs logistic regression with a penalization term.

    :param y: np.array with the labels
    :param tx: np.array with the features
    :param w: np.array with the weights
    :returns:
        loss: float, the penalized regression loss
        logistic_gradient: np.array of the penalized regression gradient
    """
    loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_logistic_gradient(y, tx, w) + lambda_ * w
    return loss, gradient
