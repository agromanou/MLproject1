#!/usr/bin/env python3

import numpy as np


def get_jet_data_split(y, tx, jet):
    """
    Given a group (jet number) returns the subset of y and tx
    corresponding to that group
    :param y: np.array with labels
    :param tx: np.array with features
    :param jet: int, indicating jet number
    :return:
        y_sub: np.array with the labels of the input jet
        tx_sub: np.array with the features of the input jet
    """
    inds = np.where(tx[:, 22] == jet)
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
        :return:
            tx: np.array with the cleaned features
        """
        tx = np.where(tx == -999, np.NaN, tx)
        tx = tx[:, ~np.all(np.isnan(tx), axis=0)]
        return tx

    def remove_constant_features(self, tx):
        """
        Removes columns that have zero variance

        :param tx: np.array with the features
        :returns:
            tx: np.array with the cleaned features
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
        :return:
            tx:  np.array with the features without outliers
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
        extra column is created indicating where the values were imputed.

        :param tx: np.array with the features
        :return:
            tx: np.array with the features without missing values
            tx_imputed: np.array with ones where values were imputed
        """
        iqr = self.q3 - self.q1
        inds = np.where(np.isnan(tx))
        tx[inds] = np.take(self.q3 + 1.5 * iqr, inds[1])

        # Creates dummies for imputed values
        tx_imputed = np.zeros((tx.shape[0], tx.shape[1]))
        array_one = np.ones(tx.shape[1])
        tx_imputed[inds] = np.take(array_one, inds[1])
        # Removes where no values were imputed
        tx_imputed = tx_imputed[:, ~np.all(tx_imputed[1:] == tx_imputed[:-1], axis=0)]
        return tx, tx_imputed

    def standardize(self, tx):
        """
        standardization robust to outliers. The median is subtracted from each
        feature and the values are divided by the interquartile range.

        :param tx: np.array with the features
        :return:
            robust: np.array with the standardized features
        """
        iqr = self.q3 - self.q1
        robust = (tx - self.median) / iqr
        return robust

    def fit_transform(self, tx):
        """
        It computes the percentiles and the median for the data sample.
        Fits the values with the training set and saves the values to be used
        with the testing set.

        :param tx: np.array with the features
        :return:
            tx: np.array with the features transformed
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
        :return:
            tx: np.array with the features transformed
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
        self.degree = None
        self.mean = None
        self.std = None

    def create_poly_features(self, tx, degree):
        """
        Build a polynomial of a certain degree

        :param tx: np.array with the features
        :param degree: int, the degree of the polynomial to be applied to each feature
        :return:
            poly: np.array with the expanded features
        """
        features = tx.shape[1]
        # Create powers for each of the features
        poly = np.ones((len(tx), 1))
        for feat in range(features):
            for deg in range(1, degree + 1):
                poly = np.c_[poly, np.power(tx[:, feat], deg)]

        poly = np.delete(poly, 0, 1)

        poly = self.build_interactions(tx, poly)
        return poly

    def build_interactions(self, tx, poly):
        """
        Build interactions between features sum, product and square of product

        :param tx: np.array of the original features
        :param poly: np.array with the features with polynomial feature expansion applied to them
        :return:
            interactions: np.array with the augmented features
        """
        features = tx.shape[1]
        for feat1 in range(features):
            for feat2 in range(feat1 + 1, features):
                poly = np.c_[poly, tx[:, feat1] + tx[:, feat2],
                                     tx[:, feat1] * tx[:, feat2],
                                     np.power(tx[:, feat1] * tx[:, feat2], 2)]
        return poly

    def standardize(self, tx):
        """
        For optimization purposes, after the transformations, all the variables
        are standardized using the mean.

        :param tx: np.array with the features
        :return
            tx: np.array with the standardized features
        """
        tx = (tx - self.mean) / self.std
        return tx

    def fit_transform(self, tx, degree):
        """
        Fits the values with the training set and saves the values to be used
        with the testing set
        Transforms the training set by creating polynomial features and
        interactions.

        :param tx: np.array with the features
        :param degree: int, maximum degree for the polynomial expansion
        """
        self.degree = degree

        tx_poly = self.create_poly_features(tx, self.degree)

        self.mean = np.mean(tx_poly, axis=0)
        self.std = np.std(tx_poly, axis=0)

        tx_standardized = self.standardize(tx_poly)
        return tx_standardized

    def transform(self, tx):
        """
        Transforms the testing set the same way the training set was transformed
        Transforms the training set.
        :param tx: np.array with the features
        :returns
            tx: np.array with the features transformed
        """
        tx_poly = self.create_poly_features(tx, self.degree)
        tx_normalized = self.standardize(tx_poly)
        return tx_normalized
