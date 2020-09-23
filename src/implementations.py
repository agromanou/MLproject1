'''
file containing ML Methods implementations

commenting style: https://www.programiz.com/python-programming/docstrings
'''

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
    return 0

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
    return 0

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
    return 0

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
    return 0

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
    return 0

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
    '''
    return 0
