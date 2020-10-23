import numpy as np

def get_error_vector(y, tx, w):
    """
    Computes the error vector
    Args:
        y: labels 
        tx: features
        w: weights
    Returns:
        error_vector: the error vector defined as y - tx.dot(w)
    """
    return y - tx.dot(w)

def get_mse(error_vector):
    """
    Computes the mse for a given error vector.
    Args:
        error_vector: error vector
    Returns:
        mse: mean squared error
    """
    return np.mean(error_vector ** 2) / 2.

def get_gradient(tx, error_vector):
    """
    Computes the gradient for a given error vector.
    Args:
        y: labels
        error_vector: error vector
    Returns:
        gradient: the gradient vector
    """
    return - tx.T.dot(error_vector) / float(error_vector.size)
    
def sigmoid(t):
    """
    Applies the sigmoid function to its input
    Args:
        t: the input
    Returns:
        sigmoid_t: sig(t)
    """
    return 1. / (1. + np.exp(-t))

def get_logistic_loss(y, tx, w):
    """
    Computes the logistic loss
    Args:
        y: labels 
        tx: features
        w: weights
    Returns:
        loss: logistic loss
    """
    inner = tx.dot(w)
    return np.sum(np.log(1. + np.exp(inner)) - y * inner)

def get_logistic_gradient(y, tx, w):
    """
    Computes the gradient of the logistic regression loss function.
    Args:
        y: labels 
        tx: features
        w: weights
    Returns:
        logistic_gradient: the gradient
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y)

def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Performs logistic regression with a penalization term
    Args:
        y: labels 
        tx: features
        w: weights
    Returns:
        loss: the penalized regression loss
        logistic_gradient: the penalized regression gradient
    """
    loss = get_logistic_loss(y, tx, w) + (lambda_ / 2.) * w.T.dot(w)
    gradient = get_logistic_gradient(y, tx, w) + lambda_ * w
    return loss, gradient
