'''
file containing ML Methods implementations

commenting style: https://www.programiz.com/python-programming/docstrings
'''
import numpy as np

from utils import *

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
    # build tx
    w = initial_w

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
