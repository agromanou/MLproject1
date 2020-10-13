#!/usr/bin/env python3

"""
Module description
"""
import numpy as np
import os

from src import DATA_DIR


class DataLoader:

    def __init__(self, sub_sample=0):
        self.tx, self.y, self.test, self.ids = self._load_data(sub_sample)

    def get_datasets(self):
        """

        :return:
        """
        data = {
            'train': {
                'tx': self.tx,
                'y': self.y
            },
            'test': {
                'tx': self.test,
            },
        }

        return data

    @staticmethod
    def _load_data(sub_sample=0):
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

        return tx, yb, test, ids

    def compute_statistics(self):
        """

        :return:
        """
        pass
