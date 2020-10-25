#!/usr/bin/env python3

"""
Module description
"""
import numpy as np
from implementations import reg_logistic_regression

def get_jet_data_split(y, tx, group):
    """
    Given a group (jet number) returns the subset of y and tx
    corresponding to that group
    :param y: np.array with labels
    :param tx: np.array with features
    :param group: int indicating jet number
    :return:
    """
    inds = np.where(tx[:, 22] == group)
    tx_sub = tx[inds]
    y_sub = y[inds]
    return y_sub, tx_sub


class DataCleaning:
    """
    This class is responsible for cleaning the data.
    The training set is meant to fit the global values (q1, q3, median, etc)
    to be then used to transform the training and testing set.
    """

    def __init__(self):
        self.q1 = None
        self.q3 = None
        self.median = None
        self.constant_columns = None

    def remove_empty_features(self, tx):
        """
        Finds the empty features presented in the dataset and removes them.

        :param tx: np.array with the features
        :return tx: np.array with the cleaned features
        """
        tx = np.where(tx == -999, np.NaN, tx)
        tx = tx[:, ~np.all(np.isnan(tx), axis=0)]
        return tx

    def remove_constant_features(self, tx):
        """
        Removes columns that have zero variance

        :param tx: np.array with the features
        :return tx: np.array with the cleaned features
        """
        if self.constant_columns is None:
            col_std = np.nanstd(tx, axis=0)
            constant_ind = np.where(col_std == 0)[0]
            self.constant_columns = constant_ind
        tx = np.delete(tx, self.constant_columns , axis=1)
        return tx

    def treat_outliers(self, tx):
        """
        Outliers are replaced by the upper or lower arm of a boxplot.
        Every value above the 75th quantile plus 1.5 times the interquartile range,
        is replaced by the 75th quantile times plus 1.5 times the interquartile range.

        Every value above the 25th quantile minus 1.5 times the interquartile range,
        is replaced by the 25th quantile minus 1.5 times the interquartile range.

        :param tx:  np.array with the features
        :return tx:  np.array with the features without outliers
        """
        iqr = self.q3 - self.q1

        outliers = np.where(tx > self.q3 + 1.5 * iqr)
        tx[outliers] = np.take(self.q3 + 1.5 * iqr, outliers[1])
        outliers = np.where(tx < self.q1 - 1.5 * iqr)
        tx[outliers] = np.take(self.q1 + 1.5 * iqr, outliers[1])
        return tx

    def treat_missing_data(self, tx):
        """
        Separating by jets, there is only one column with missing values
        that may be undefined if the topology of the event is too far from the
        expected topology.
        The missing values were replaced by the upper arm of the boxplot and an
        extra column is created indicating where the values were imputated.

        :param tx: np.array with the features
        :return tx: np.array with the features without missing values
        :return tX_imputed: np.array with where values were imputated
        """
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
        """
        standardization robust to outliers. The median is subtracted from each
        feature and the values are divided by the interquartile range.

        :param tx: np.array with the features
        :return robust: np.array with the standardized features
        """
        iqr = self.q3 - self.q1
        robust = (tx - self.median) / iqr
        return robust

    def fit_transform(self, tx):
        """
        It computes the percentiles and the median for the data sample.
        Fits the values with the trainig set and saves the values to be used
        with the testing set
        Transforms the training set.
        :param tx: np.array with the features
        :param tx: np.array with the features transformed
        """
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
        Transforms the testing set the same way the training set was transformed
        Transforms the training set.
        :param tx: np.array with the features
        :param tx: np.array with the features transformed
        """
        tx = self.remove_empty_features(tx)
        tx = self.remove_constant_features(tx)
        tx = self.treat_outliers(tx)
        tx, tx_imputed = self.treat_missing_data(tx)
        tx = self.standardize(tx)
        tx = np.c_[tx, tx_imputed]
        return tx


class FeatureEngineering:
    """
    This class is responsible for engineering new features
    The training set is meant to fit the global values (mean, std)
    to be then used to transform the training and testing set.
    """
    def __init__(self):
        self.top_features_list = None
        self.degree = None
        self.number_interactions = None
        self.mean = None
        self.std = None

    # def create_poly_features(self, tX, degree):
    #     """
    #     Creates polynomial expansion
    #
    #     :param tX: np.array with the features
    #     :param degree: int, maximum number of degree desired for the polynomial
    #     expansion
    #     :return tX_poly: np.array with the expanded features
    #     """
    #     poly = np.ones((len(tX), 1))
    #     for deg in range(1, degree+1):
    #         poly = np.c_[poly, np.power(tX, deg)]
    #
    #     tX_poly = np.delete(poly, 0, 1)
    #     return tX_poly

    def create_poly_features(self, tX, degree):
        """
        Build a polynomial of a certain degree with crossed terms (applying sum, product and square of product)

        :param tX: Features
        :param degree: Degree of the polynomial (for each individual feature)
        :return: poly: Expanded features
        """
        features = tX.shape[1]
        # Create powers for each of the features
        poly = np.ones((len(tX), 1))
        for feat in range(features):
            for deg in range(1, degree + 1):
                poly = np.c_[poly, np.power(tX[:, feat], deg)]

        poly = np.delete(poly, 0, 1)

        poly = self.build_interractions(tX, poly)
        return poly

    def build_interractions(self, tX, poly):
        """
        Build interractions between features sum, product and square of product

        :param tX: Features
        :return: poly: Expanded features
        """
        features = tX.shape[1]
        for feat1 in range(features):
            for feat2 in range(feat1 + 1, features):
                poly = np.c_[poly, tX[:, feat1] + tX[:, feat2],
                             tX[:, feat1] * tX[:, feat2],
                             np.power(tX[:, feat1] * tX[:, feat2], 2)]
        return poly

    def select_top_vars(self, tX, y, n=5):
        """
        Selects the variables to create interactions with. The selected features
        are the ones with the largest (in magnitude) coefficients

        :param tX: np.array with the features
        :param y: Lables corresponding to tX
        :param n: Number of top features to be returned
        :return top_features_list: np.array with top features
        """
        initial_w =  np.zeros((tX.shape[1]))
        w, loss = reg_logistic_regression(y, tX, 1e-6, initial_w, 1000, 1e-6)
        top_features_list = np.argsort(-abs(w))[:int(n)]
        return top_features_list

    def standardize(self, tX):
        """
        For optimization purposes, after the transformations, all the variables
        are standardized using the mean.

        :param tX: np.array with the features
        :return tX: np.array with the standardized features
        """
        tX = (tX- self.mean)/(self.std)
        return tX

    def fit_transform(self, tX, y, degree, num_top_vars):
        """
        Fits the values with the trainig set and saves the values to be used
        with the testing set
        Transforms the training set by creating polynomial features and
        interactions.

        :param tX: np.array with the features
        :param y: np.array with the labels
        :param degree: int, maximum degree for the polynomial expansion
        :param num_top_vars: int, number of features to create interactions with
        :param tX_normalized: np.array with the features transformed
        """
        self.degree = degree
        self.num_top_vars = num_top_vars

        tX_poly =  self.create_poly_features(tX, degree)

        self.mean = np.mean(tX_poly, axis = 0)
        self.std = np.std(tX_poly, axis = 0)

        tX_normalized =self.standardize(tX_poly)
        return tX_normalized


    def transform(self, tX):
        """
        Transforms the testing set the same way the training set was transformed
        Transforms the training set.
        :param tx: np.array with the features
        :param tx: np.array with the features transformed
        """
        tX_poly =  self.create_poly_features(tX, self.degree)
        tX_normalized =self.standardize(tX_poly)
        return tX_normalized
