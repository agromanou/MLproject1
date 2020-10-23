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
