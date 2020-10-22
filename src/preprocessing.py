#!/usr/bin/env python3

"""
Module description
"""
import numpy as np
from models import Models


def get_jet_data_split(y, tx, group):
    inds = np.where(tx[:, 22] == group)
    tx_sub = tx[inds]
    y_sub = y[inds]
    return y_sub, tx_sub


class DataCleaning:
    def __init__(self):
        self.q1 = None
        self.q3 = None
        self.median = None
        self.constant_columns = None

    def remove_empty_features(self, tx):
        """
        Finds the empty features presented in the dataset and removes them.

        :param tx: a numpy array representing the given features
        :return: a numpy array representing the cleaned given features
        """
        tx = np.where(tx == -999, np.NaN, tx)
        tx = tx[:, ~np.all(np.isnan(tx), axis=0)]
        return tx


    def remove_constant_features(self, tx):
        """

        :param tx: a numpy array representing the given features
        :return: a numpy array representing the cleaned given features
        """
        if self.constant_columns is None:
            col_std = np.nanstd(tx, axis=0)
            constant_ind = np.where(col_std == 0)[0]
            self.constant_columns = constant_ind
        tx = np.delete(tx, self.constant_columns , axis=1)
        return tx


    def treat_outliers(self, tx):
        iqr = self.q3 - self.q1

        outliers = np.where(tx > self.q3 + 1.5 * iqr)
        tx[outliers] = np.take(self.q3 + 1.5 * iqr, outliers[1])
        outliers = np.where(tx < self.q1 - 1.5 * iqr)
        tx[outliers] = np.take(self.q1 + 1.5 * iqr, outliers[1])
        return tx

    def treat_missing_data(self, tx):
        # Fill na with maximum value
        iqr = self.q3 - self.q1
        inds = np.where(np.isnan(tx))
        tx[inds] = np.take(self.q3 + 1.5 * iqr, inds[1])

        # Creates dummies for imputed values
        tX_imputed = np.zeros((tx.shape[0], tx.shape[1]))
        array_one = np.ones(tx.shape[1])
        tX_imputed[inds] = np.take(array_one, inds[1])
        # Removes where no values were imputed
        tX_imputed = tX_imputed[:, ~np.all(tX_imputed[1:] == tX_imputed[:-1],
                                                                    axis=0)]
        return tx, tX_imputed

    def standardize(self, tx):
        iqr = self.q3 - self.q1
        return (tx - self.median) / iqr

    def fit_transform(self, tx):
        """
        It computes the percentiles and the median for the given data sample.

        :param tx: np.array, with the data
        """
        # Robust standardization & outliers
        tx = self.remove_empty_features(tx)
        tx = self.remove_constant_features(tx)
        self.q1 = np.nanpercentile(tx, q=25, axis=0)
        self.median = np.nanpercentile(tx, q=50, axis=0)
        self.q3 = np.nanpercentile(tx, q=75, axis=0)

        tx = self.treat_outliers(tx)
        tx, tx_imputed = self.treat_missing_data(tx)
        tx = self.standardize(tx)

        tx = np.c_[tx, tx_imputed]
        return tx

    def transform(self, tx):
        """
        :param tx: np.array, with the data
        :return: np.array with the standardized data
        """
        tx = self.remove_empty_features(tx)
        tx = self.remove_constant_features(tx)
        tx = self.treat_outliers(tx)
        tx, tx_imputed = self.treat_missing_data(tx)
        tx = self.standardize(tx)
        tx = np.c_[tx, tx_imputed]
        return tx



class FeatureEngineering:
    def __init__(self):
        self.top_features_list = None
        self.degree = None
        self.number_interactions = None
        self.mean = None
        self.std = None

    def create_poly_features(self, tX, degree):
        poly = np.ones((len(tX), 1))
        for deg in range(1, degree+1):
            poly = np.c_[poly, np.power(tX, deg)]

        tX_poly = np.delete(poly, 0, 1)
        return tX_poly

    def create_interactions(self, tX, columns):
        for col1 in columns:
            for col2 in columns:
                if col1 > col2:
                    col3 = np.multiply(tX[:, col1], tX[:, col2])
                    tX = np.c_[tX, col3]
        return tX

    def select_top_vars(self, tX, y, n=5):
        initial_w =  np.zeros((tX.shape[1]))
        model = Models()
        w, loss = model.reg_logistic_regression(y, tX, 1e-7, initial_w, 1000, 1e-8)
        top_features_list = np.argsort(-abs(w))[:int(n)]
        return top_features_list


    def normalize(self, tX):
        return (tX- self.mean)/(self.std)

    def fit_transform(self, tX, y, degree, num_top_vars):
        """
        To be used with the training set

        :param tx: np.array, with the data
        """
        self.degree = degree
        self.num_top_vars = num_top_vars

        tX_poly =  self.create_poly_features(tX, degree)
        top_features_list = self.select_top_vars(tX_poly, y, num_top_vars)
        self.top_features_list = top_features_list
        tX_interactions = self.create_interactions(tX_poly, top_features_list)

        self.mean = np.mean(tX_interactions, axis = 0)
        self.std = np.std(tX_interactions, axis = 0)

        tX_normalized =self.normalize(tX_interactions)
        return tX_normalized

    # To be used with the testing set
    def transform(self, tX):
        """
        :param tx: np.array, with the data
        :return: np.array with the new features
        """
        tX_poly =  self.create_poly_features(tX, self.degree)
        tX_interactions = self.create_interactions(tX_poly, self.top_features_list)
        tX_normalized =self.normalize(tX_interactions)
        return tX_normalized
