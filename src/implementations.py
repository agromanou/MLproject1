#!/usr/bin/env python3

"""
Module description
"""
import numpy as np


class Models:

    def __init__(self, tx_train=None, y_train=None, tx_val=None, y_val=None):
        """
        This class represents a logistic regression model.

        :param tx_train: np. array of the features of the training set
        :param y_train: np. array of the labels of the training set
        :param tx_val: np. array of the features of the validation set
        :param y_val: np. array of the labels of the validation set
        """
        self.tx_train = tx_train
        self.y_train = y_train

        self.tx_val = tx_val
        self.y_val = y_val

        self.w = None
        self.lambda_ = None

    def fit(self, gamma, initial_w, lambda_=0, epochs=1000, verbose=False):
        """
        Fit the model to the data.

        :param gamma: float, the learning rate of the training
        :param initial_w: np array with initial weights
        :param lambda_: float, the penalization term
        :param epochs: int, the number of iterations of the training
        :param verbose: boolean, if true the full training logging is printed

        :return:
            loss_train: np array with the training loss values for each epoch
            loss_val: np array with the validation loss values for each epoch
        """

        loss_train = list()
        loss_val = list()

        ws = [initial_w]
        w = initial_w
        self.lambda_ = lambda_
        threshold = 1e-8

        for epoch in range(epochs):

            loss, w = self.learning_by_penalized_gradient(self.y_train, self.tx_train,
                                                                w, gamma,lambda_)
            val_loss = self.calculate_penalized_loss_log(self.y_val,
                                                        self.tx_val, w,lambda_)

            ws.append(w)
            loss_train.append(loss / len(self.tx_train))
            loss_val.append(val_loss / len(self.tx_val))

            if len(loss_train) > 1 and np.abs(loss_train[-1] - loss_train[-2]) < threshold:
                break  # convergence criterion met

            print("Gradient Descent({epoch}/{epochs}): "
                  "training-loss={loss:.5f} | validation-loss={val_loss:.5f}".format(
                epoch=epoch, epochs=epochs - 1, loss=loss, val_loss=val_loss)) if verbose else None

        self.w = w

        return loss_train, loss_val

    def score(self, tx, y):
        """
        It applies the trained model to the given data and computes the loss based on the given labels.

        :param tx: np. array of the features
        :param y: np. array of the labels
        :return: float, the loss of the model on the given dataset
        """
        loss = self.calculate_penalized_loss_log(y, tx, self.w)
        return loss

    def predict(self, tx):
        """
        It applies the training weights and return the predictions for a given dataset.

        :param tx: np. array of the features
        :return: np.array with the predicted labels
        """
        threshold = 0.5

        y_pred = np.dot(tx, self.w)
        y_pred[np.where(y_pred <= threshold)] = 0
        y_pred[np.where(y_pred > threshold)] = 1

        return y_pred

    @staticmethod
    def sigmoid(t):
        """
        Applies the sigmoid function on the given data.
        """
        return 1.0 / (1 + np.exp(-t))

    def calculate_penalized_loss_log(self, y, tx, w):
        """
        Computes the cost by penalized negative log likelihood.
        """
        pred = self.sigmoid(tx.dot(w))
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        return np.squeeze(- loss) + self.lambda_ * np.squeeze(w.T.dot(w))

    def calculate_gradient_log(self, y, tx, w):
        """
        Computes the gradient of loss.
        """
        pred = self.sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y)
        return grad


    def calculate_penalized_loss_log(self, y, tx, w, lambda_):
        """
        Computes the cost by penalized negative log likelihood.
        """
        pred = self.sigmoid(tx.dot(w))
        loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
        return np.squeeze(- loss) + lambda_ * np.squeeze(w.T.dot(w))

    def calculate_gradient_log(self, y, tx, w):
        """
        Computes the gradient of loss.
        """
        pred = self.sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y)
        return grad

    def learning_by_penalized_gradient(self, y, tx, w, gamma, lambda_):
        """
        Does one step of gradient descent, using the penalized logistic regression.
        Returns the loss and updated w.
        """
        loss = self.calculate_penalized_loss_log(y, tx, w, lambda_)
        gradient = self.calculate_gradient_log(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

        return loss, w

    ## Other models and implementations
    def calculate_mse(self,e):
        """Calculate the mse for vector e."""
        return 1/2*np.mean(e**2)


    def calculate_mae(self,e):
        """Calculate the mae for vector e."""
        return np.mean(np.abs(e))


    def compute_loss(self,y, tx, w):
        """Calculate the loss.

        You can calculate the loss using mse or mae.
        """
        e = y - tx.dot(w)
        return self.calculate_mse(e)

    def least_squares_GD(self, y, tx, initial_w, max_iters, gamma):
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
            grad, err = self.compute_gradient(y, tx, w)
            loss = self.calculate_mse(err)
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


    def least_squares_SGD(self,y, tx, initial_w, max_iters, gamma):
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
                grad, _ = self.compute_stoch_gradient(y_batch, tx_batch, w)
                # update w through the stochastic gradient update
                w = w - gamma * grad
                # calculate loss
                loss = self.compute_loss(y, tx, w)
                # store w and loss
                ws.append(w)
                losses.append(loss)

            print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

        loss = losses[-1]
        w = ws[-1]
        return w, loss


    def least_squares(self, y, tx):
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
        loss = self.compute_loss(y, tx, w)
        return w, loss


    def ridge_regression(self,y, tx, lambda_):
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
        loss = self.compute_loss(y, tx, w)
        return w, loss


    def logistic_regression(self,y, tx, initial_w, max_iters, gamma):
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
            loss, w = self,learning_by_gradient_descent(y, tx, w, gamma)
            # log info
            if iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            # converge criterion
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break

        print("loss={l}".format(l=self.calculate_loss_log(y, tx, w)))
        loss = losses[-1]
        return w, loss


    def reg_logistic_regression(self,y, tx, lambda_ , initial_w, max_iters, gamma):
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
        threshold = 1e-8
        w = initial_w
        ws = [initial_w]
        losses = []
        w = initial_w
        for iter in range(max_iters):
            loss, w = self.learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
            # log info
            if iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=iter, l=loss))

            ws.append(w)
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break # convergence criterion met
        return ws[-1], losses[-1]
