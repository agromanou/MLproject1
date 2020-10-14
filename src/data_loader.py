#!/usr/bin/env python3

"""
Module description
"""
import numpy as np
import os

from src import DATA_DIR


class DataLoader:

    def __init__(self, sub_sample=0):
        self.tx, self.y, self.tx_te, self.y_te, self.test, self.ids = self._load_data(sub_sample)

    def get_datasets(self):
        """

        :return:
        """
        data = {
            'train': {
                'tx': self.tx,
                'y': self.y
            },
            'test_labeled': {
                'tx_te': self.tx_te,
                'y_te': self.y_te
            },
            'test': {
                'tx': self.test,
            },
        }

        return data

    def _load_data(self, sub_sample=0):
        """
        Loads the train and test data needed.

        :param sub_sample: int, the size of a sample we want to take from the data
        :return: tX (features), y (class labels), ids (event ids), test (test data)
        """
        train_path_dataset = os.path.join(DATA_DIR, 'train.csv')
        test_path_dataset = os.path.join(DATA_DIR, 'test.csv')

        tx = np.genfromtxt(train_path_dataset, delimiter=",", skip_header=1)
        y = np.genfromtxt(train_path_dataset, delimiter=",", skip_header=1, dtype=str, usecols=1)
        test = np.genfromtxt(test_path_dataset, delimiter=",", skip_header=1)

        ids = tx[:, 0].astype(np.int)  # get ids
        tx = tx[:, 2:]  # get train data minus the ids and labels
        test = test[:, 2:]  # get test data minus the ids and labels

        # convert class labels from strings to binary (0,1)
        yb = np.ones(len(y))
        yb[np.where(y == 'b')] = 0

        # sub-sample
        if sub_sample:
            yb = yb[::50]
            tx = tx[::50]
            ids = ids[::50]
        
        tx, tx_te, yb, yb_te = self.split_data(tx, yb, 0.9)

        return tx, yb, tx_te, yb_te, test, ids

    def compute_statistics(self):
        """

        :return:
        """
        pass

    @staticmethod
    def split_data(x, y, ratio, myseed=1):
        """split the dataset based on the split ratio."""
        # set seed
        np.random.seed(myseed)
        # generate random indices
        num_row = len(y)
        indices = np.random.permutation(num_row)
        index_split = int(np.floor(ratio * num_row))
        index_tr = indices[: index_split]
        index_te = indices[index_split:]
        # create split
        x_tr = x[index_tr]
        x_te = x[index_te]
        y_tr = y[index_tr]
        y_te = y[index_te]
        return x_tr, x_te, y_tr, y_te