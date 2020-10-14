#!/usr/bin/env python3

"""
Module description
"""
import numpy as np
from src.implementations import \
    learning_by_penalized_gradient, \
    penalized_logistic_regression, \
    calculate_loss_log


class Model:

    def __init__(self, tx_train, y_train, tx_val, y_val):
        self.tx_train = tx_train
        self.y_train = y_train

        self.tx_val = tx_val
        self.y_val = y_val

        self.w = None
        self.lambda_ = None

    def fit(self, gamma, initial_w, lambda_=0, epochs=1000):
        """
        Fit the model to the data

        :param gamma:
        :param initial_w:
        :param lambda_:
        :param epochs:
        :return:
        """

        loss_train = list()
        loss_val = list()

        ws = [initial_w]
        w = initial_w
        self.lambda_ = lambda_

        for epoch in range(epochs):

            loss, w = learning_by_penalized_gradient(self.y_train, self.tx_train, w, gamma, lambda_)
            val_loss = calculate_loss_log(self.y_val, self.tx_val, w)

            ws.append(w)
            loss_train.append(loss)
            loss_val.append(val_loss)

            print("Gradient Descent({epoch}/{epochs}): training-loss={loss} | validation-loss={val_loss}".format(
                epoch=epoch, epochs=epochs - 1, loss=loss, val_loss=val_loss))

        self.w = w

        return loss_train, loss_val

    def _train(self, tx, y):
        raise NotImplementedError('Please call a specific Model type of class.')

    def score(self, tx, y):
        loss, _ = penalized_logistic_regression(y, tx, self.w, self.lambda_)
        return loss

    def predict(self, x_test):
        raise NotImplementedError('Not implemented')

    def compute_test_loss(self):
        pass


class LogisticRegression(Model):

    def __init__(self, tx, y, **kwargs):
        super().__init__(tx, y)

    def _train(self, tx, y):
        pass
