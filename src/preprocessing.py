#!/usr/bin/env python3

"""
Module description
"""
import numpy as np


def get_jet_data_split(y, tx, jet_num):
    """
    Splits the given data set such that only the data points with a certain
    jet number remains, where jet number is a discrete valued feature. In
    other words, filters the data set using the jet number.

    :param y: a numpy array representing the given labels
    :param tx: a numpy array representing the given features
    :param jet_num: the certain value of the discrete feature jet number
    :return:
        y_masked: numpy array of labels of data points having the specified jet number
        tx_masked: numpy array of features of data points having the specified jet number
        ids_masked: numpy array of ids of data points having the specified jet number
    """
    mask = tx[:, 22] == jet_num
    return y[mask], tx[mask]


def remove_empty_features(tx):
    """
    Finds the empty features presented in the dataset and removes them.

    :param tx: a numpy array representing the given features
    :return: a numpy array representing the cleaned given features
    """
    tx = np.where(tx == -999, np.NaN, tx)
    tx = tx[:, ~np.all(np.isnan(tx), axis=0)]

    return tx


def remove_constant_features(tx):
    """

    :param tx: a numpy array representing the given features
    :return: a numpy array representing the cleaned given features
    """
    col_std = np.nanstd(tx, axis=0)
    constant_ind = np.where(col_std == 0)[0]
    return np.delete(tx, constant_ind, axis=1)


def treat_missing_data(tx):
    # Fill na with median (in this case is 0)
    median = np.zeros(tx.shape[1])
    inds = np.where(np.isnan(tx))
    tx[inds] = np.take(median, inds[1])

    # Creates dummies for imputed values
    tX_imputed = np.zeros((tx.shape[0], tx.shape[1]))
    array_one = np.ones(tx.shape[1])
    tX_imputed[inds] = np.take(array_one, inds[1])

    return tx, tX_imputed


class Standardization:
    def __init__(self):
        self.q1 = None
        self.q3 = None
        self.median = None

    def fit(self, tx):
        """
        It computes the percentiles and the median for the given data sample.

        :param tx: np.array, with the data
        """
        # Robust standardization & outliers
        self.q1 = np.nanpercentile(tx, q=25, axis=0)
        self.median = np.nanpercentile(tx, q=50, axis=0)
        self.q3 = np.nanpercentile(tx, q=75, axis=0)

    def transform(self, tx):
        """

        :param tx: np.array, with the data
        :return: np.array with the standardized data
        """
        iqr = self.q3 - self.q1

        outliers = np.where(tx > self.q3 + 1.5 * iqr)
        tx[outliers] = np.take(self.q3 + 1.5 * iqr, outliers[1])
        outliers = np.where(tx < self.q1 - 1.5 * iqr)
        tx[outliers] = np.take(self.q1 + 1.5 * iqr, outliers[1])

        return (tx - self.median) / iqr