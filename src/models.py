#!/usr/bin/env python3

"""
Module description
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.implementations import learning_by_penalized_gradient, penalized_logistic_regression


class Model:

    def __init__(self, tx, y):
        self.tx = tx
        self.y = y

        self.w = None
        self.lambda_ = None

    def fit(self, gamma, initial_w, lambda_=0, max_iter=1000, folds=0):
        """
        Fit the model to the data

        :param gamma:
        :param initial_w:
        :param lambda_:
        :param max_iter:
        :param folds:
        :return:
        """

        # TODO: cross validation
        tx_train = self.tx[:80]
        tx_val = self.tx[80:100]

        y_train = self.y[:80]
        y_val = self.y[80:100]

        loss_train = list()
        loss_val = list()

        ws = [initial_w]
        w = initial_w
        self.lambda_ = lambda_

        for n_iter in range(max_iter):

            loss, w = learning_by_penalized_gradient(y_train, tx_train, w, gamma, lambda_)

            ws.append(w)
            loss_train.append(loss)

            # TODO: apply w to val set

            print("Gradient Descent({bi}/{ti}): training-loss={loss}".format(
                bi=n_iter, ti=max_iter - 1, loss=loss))

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
