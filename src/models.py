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

    def fit(self, gamma, initial_w, lambda_=0, epochs=1000, batch_size=0, verbose=False):
        """
        Fit the model to the data.

        :param gamma: float, the learning rate of the training
        :param initial_w: np array with initial weights
        :param lambda_: float, the penalization term
        :param epochs: int, the number of iterations of the training
        :param verbose: boolean, if true the full training logging is printed
        :param batch_size: int, if 0 it runs GD

        :return:
            loss_train: np array with the training loss values for each epoch
            loss_val: np array with the validation loss values for each epoch
        """

        loss_train = list()
        loss_val = list()

        ws = [initial_w]
        w = initial_w
        threshold = 1e-8

        if not batch_size:
            batch_size = len(self.y_train)

        for epoch in range(epochs):
            for y_batch, tx_batch in self.batch_iter(self.y_train, self.tx_train, batch_size=batch_size, num_batches=1):
                loss, w = self.learning_by_penalized_gradient(y_batch, tx_batch, w, gamma, lambda_)
                val_loss = self.calculate_penalized_loss_log(self.y_val, self.tx_val, w, lambda_)

                ws.append(w)
                loss_train.append(loss / len(tx_batch))
                loss_val.append(val_loss / len(self.tx_val))

                if len(loss_train) > 1 and np.abs(loss_train[-1] - loss_train[-2]) < threshold:
                    break  # convergence criterion met

                print("\t\tGradient Descent({epoch}/{epochs}): "
                      "training-loss={loss:.3f} | validation-loss={val_loss:.3f}".format(
                    epoch=epoch, epochs=epochs - 1, loss=loss / len(tx_batch),
                    val_loss=val_loss / len(self.tx_val))) if verbose else None

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

        y_pred = self.sigmoid(np.dot(tx, self.w))
        y_pred[np.where(y_pred <= threshold)] = 0
        y_pred[np.where(y_pred > threshold)] = 1

        return y_pred

    @staticmethod
    def sigmoid(t):
        """
        Applies the sigmoid function on the given data.
        """
        return 1.0 / (1 + np.exp(-t))

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

    def learning_by_penalized_gradient(self, y, tx, w, gamma, lambda_):
        """
        Does one step of gradient descent, using the penalized logistic regression.
        Returns the loss and updated w.
        """
        loss = self.calculate_penalized_loss_log(y, tx, w, lambda_)
        gradient = self.calculate_gradient_log(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

        return loss, w

    def reg_logistic_regression(self, y, tx, lambda_, initial_w, max_iters, gamma, verbose=False):
        '''
        regularized logistic regression using gradient descentor
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
        ws = [initial_w]
        losses = []
        w = initial_w
        for iter in range(max_iters):
            loss, w = self.learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
            # log info
            if iter % 100 == 0:
                print("Current iteration={i}, loss={l:.3f}".format(i=iter, l=loss)) if verbose else None

            ws.append(w)
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break  # convergence criterion met
        return ws[-1], losses[-1]

    @staticmethod
    def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
        """
        Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
        Example of use :
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
            <DO-SOMETHING>
        """
        data_size = len(y)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_y = y[shuffle_indices]
            shuffled_tx = tx[shuffle_indices]
        else:
            shuffled_y = y
            shuffled_tx = tx
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            if start_index != end_index:
                yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
