'''
file containing ML Methods implementations

commenting style: https://www.programiz.com/python-programming/docstrings
'''
import numpy as np


# Gradients
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def calculate_gradient_log(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))


def calculate_loss_log(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss_log(y, tx, w)
    grad = calculate_gradient_log(y, tx, w)
    w -= gamma * grad
    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    #     loss = compute_logistic_loss(y, tx, w) + (lambda_ / 2) * w.T.dot(w)
    #     gradient = compute_logistic_gradient(y, tx, w) + lambda_ * w

    loss = calculate_loss_log(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient_log(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient

    return loss, w


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''
    linear regression using gradient descent
        parameters:
            y (np.array): label or outcome
            tx (np.array): matrix of features
            initial_w (np.array): initial weight vector
            max_iters (int): number of steps to run
            gamma: step-size

        returns:
            w (np.array): last weight vector of the method
            loss (float): corresponding loss value (cost function)

    '''
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    loss = losses[-1]
    w = ws[-1]
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    '''
    linear regression using stochastic gradient descent
        parameters:
            y (np.array): label or outcome
            tx (np.array): matrix of features
            initial_w (np.array): initial weight vector
            max_iters (int): number of steps to run
            gamma: step-size

        returns:
            w (np.array): last weight vector of the method
            loss (float): corresponding loss value (cost function)
    '''
        # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    loss = losses[-1]
    w = ws[-1]
    return w, loss


def least_squares(y, tx):
    '''
    least squares regression using normal equations
        parameters:
            y (np.array): label or outcome
            tx (np.array): matrix of features

        returns:
            w (np.array): last weight vector of the method
            loss (float): corresponding loss value (cost function)

    '''
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    '''
    ridge regression using normal equations
        parameters:
            y (np.array): label or outcome
            tx (np.array): matrix of features
            _lambda (float): regularization parameter

        returns:
            w (np.array): last weight vector of the method
            loss (float): corresponding loss value (cost function)

    '''
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w =  np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''
    logistic regression using gradient descent or SGD
        parameters:
            y (np.array): label or outcome
            tx (np.array): matrix of features
            initial_w (np.array): initial weight vector
            max_iters (int): number of steps to run
            gamma: step-size

        returns:
            w (np.array): last weight vector of the method
            loss (float): corresponding loss value (cost function)
    '''
    w = initial_w
    losses = []
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    print("loss={l}".format(l=calculate_loss_log(y, tx, w)))
    loss = losses[-1]
    return w, loss


def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    '''
    regularized logistic regression using gradient descentor SGD
        parameters:
            y (np.array): label or outcome
            tx (np.array): matrix of features
            _lambda (float): regularization parameter
            initial_w (np.array): initial weight vector
            max_iters (int): number of steps to run
            gamma: step-size
        returns:
            w (np.array): last weight vector of the method
            loss (float): corresponding loss value (cost function)
    '''
    w = initial_w

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    loss = losses[-1]
    return w, loss




#===============================


import numpy as np

# The helper methods used by the learning methods above are implemented below:

def compute_error_vector(y, tx, w):
    """
    Computes the error vector that is defined as y - tx . w
    Args:
        y: labels
        tx: features
        w: weight vector
    Returns:
        error_vector: the error vector defined as y - tx.dot(w)
    """
    return y - tx.dot(w)

def compute_mse(error_vector):
    """
    Computes the mean squared error for a given error vector.
    Args:
        error_vector: error vector computed for a specific dataset and model
    Returns:
        mse: numeric value of the mean squared error
    """
    return np.mean(error_vector ** 2) / 2

def compute_gradient(tx, error_vector):
    """
    Computes the gradient for the mean squared error loss function.
    Args:
        y: labels
        error_vector: error vector computed for a specific data set and model
    Returns:
        gradient: the gradient vector computed according to its definition
    """
    return - tx.T.dot(error_vector) / error_vector.size

def build_polynomial(x, degree):
    """
    Extends the feature matrix, x, by adding a polynomial basis of the given degree.
    Args:
        x: features
        degree: degree of the polynomial basis
    Returns:
        augmented_x: expanded features based on a polynomial basis
    """
    num_cols = x.shape[1] if len(x.shape) > 1 else 1
    augmented_x = np.ones((len(x), 1))
    for col in range(num_cols):
        for degree in range(1, degree + 1):
            if num_cols > 1:
                augmented_x = np.c_[augmented_x, np.power(x[ :, col], degree)]
            else:
                augmented_x = np.c_[augmented_x, np.power(x, degree)]
        if num_cols > 1 and col != num_cols - 1:
            augmented_x = np.c_[augmented_x, np.ones((len(x), 1))]
    return augmented_x

def compute_rmse(loss_mse):
    """
    Computes the root mean squared error.
    Args:
        loss_mse: numeric value of the mean squared error loss
    Returns:
        loss_rmse: numeric value of the root mean squared error loss
    """
    return np.sqrt(2 * loss_mse)

def sigmoid(t):
    """
    Applies the sigmoid function to a given input t.
    Args:
        t: the given input to which the sigmoid function will be applied.
    Returns:
        sigmoid_t: the value of sigmoid function applied to t
    """
    return 1. / (1. + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """
    Computes the loss as the negative log likelihood of picking the correct label.
    Args:
        y: labels
        tx: features
        w: weight vector
    Returns:
        loss: the negative log likelihood of picking the correct label
    """
    tx_dot_w = tx.dot(w)
    return np.sum(np.log(1. + np.exp(tx_dot_w)) - y * tx_dot_w)


def compute_logistic_gradient(y, tx, w):
    """
    Computes the gradient of the loss function used in logistic regression.
    Args:
        y: labels
        tx: features
        w: weight vector
    Returns:
        logistic_gradient: the gradient of the loss function used in
            logistic regression.
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y)


# def penalized_logistic_regression(y, tx, w, lambda_):
#     """
#     Adds the penalization term (2-norm of w vector) on top of the normal
#     logistic loss. Computes the modified loss and gradient.
#     Args:
#         y: labels
#         tx: features
#         w: weight vector
#     Returns:
#         loss: the modified version of the normal logistic loss
#         logistic_gradient: the gradient of modified loss function used in
#             penalized logistic regression.
#     """
#     loss = compute_logistic_loss(y, tx, w) + (lambda_ / 2) * w.T.dot(w)
#     gradient = compute_logistic_gradient(y, tx, w) + lambda_ * w
#     return loss, gradient

def cross_terms(x, x_initial):
    """
    Adds the multiplication of different features as new features.
    Args:
        x: the given feature matrix
        x_initial: the features whose multiplications will be added
    Returns:
        x_cross_terms: feature matrix with cross terms
    """
    for col1 in range(x_initial.shape[1]):
        for col2 in np.arange(col1 + 1, x_initial.shape[1]):
            if col1 != col2:
                x = np.c_[x, x_initial[:, col1] * x_initial[:, col2]]
    return x

def log_terms(x, x_initial):
    """
    Adds the logarithms of features as new features.
    Args:
        x: the given feature matrix
        x_initial: the features whose logarithms will be added
    Returns:
        x_log_terms: feature matrix with logarithms
    """
    for col in range(x_initial.shape[1]):
        current_col = x_initial[:, col]
        current_col[current_col <= 0] = 1
        x = np.c_[x, np.log(current_col)]
    return x

def sqrt_terms(x, x_initial):
    """
    Adds the square roots of features as new features.
    Args:
        x: the given feature matrix
        x_initial: the features whose square roots will be added
    Returns:
        x_sqrt_terms: feature matrix with square roots
    """
    for col in range(x_initial.shape[1]):
        current_col = np.abs(x_initial[:, col])
        x = np.c_[x, np.sqrt(current_col)]
    return x

def apply_trigonometry(x, x_initial):
    """
    Adds the sin and cos of features as new features.
    Args:
        x: the given feature matrix
        x_initial: the features whose sin and cos will be added
    Returns:
        x_sqrt_terms: feature matrix with sine values
    """
    for col in range(x_initial.shape[1]):
        x = np.c_[x, np.sin(x_initial[:, col])]
        x = np.c_[x, np.cos(x_initial[:, col])]
    return x

def feature_engineering(x, degree, has_angles = False):
    """
    Builds a polynomial with the given degree from the initial features,
    add the cross terms, logarithms and square roots of the initial features
    as new features. Also includes the sine of features as an option.
    Args:
        x: features
        degree: degree of the polynomial basis
        has_angles: Boolean value to determine including sin and cos of features
    Returns:
        x_engineered: engineered features
    """
    x_initial = x
    x = build_polynomial(x, degree)
    x = cross_terms(x, x_initial)
    x = log_terms(x, x_initial)
    x = sqrt_terms(x, x_initial)
    if has_angles:
        x = apply_trigonometry(x, x_initial)
    return x


# The six compulsory learning methods are as implemented as follows:

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    Args:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        error_vector = compute_error_vector(y, tx, w)
        loss = compute_mse(error_vector)
        gradient_vector = compute_gradient(tx, error_vector)
        w = w - gamma * gradient_vector
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    Args:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        random_index = np.random.randint(len(y))
        # sample a random data point from y vector
        y_random = y[random_index]
        # sample a random row vector from tx matrix
        tx_random = tx[random_index]
        error_vector = compute_error_vector(y_random, tx_random, w)
        loss = compute_mse(error_vector)
        gradient_vector = compute_gradient(tx_random, error_vector)
        w = w - gamma * gradient_vector
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Args:
        y: labels
        tx: features
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    coefficient_matrix = tx.T.dot(tx)
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    error_vector = compute_error_vector(y, tx, w)
    loss = compute_mse(error_vector)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Args:
        y: labels
        tx: features
        lambda_: regularization parameter
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    coefficient_matrix = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    error_vector = compute_error_vector(y, tx, w)
    loss = compute_mse(error_vector)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using stochastic gradient descent
    Args:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on logistic loss
    """
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        gradient_vector = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient_vector
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, epochs, gamma):
    """
    Regularized logistic regression using gradient descent
    Args:
        y: labels
        tx: features
        lambda_: regularization parameter
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on logistic loss
    """
    threshold = 1e-8
    w = initial_w
    ws = [initial_w]
    losses = []
    w = initial_w
    for iter in range(epochs):
        loss, gradient_vector = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient_vector
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]
