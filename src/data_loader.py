#!/usr/bin/env python3

"""
Module description
"""
import numpy as np
import os


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT_DIR = get_root_dir()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')


class DataLoader:

    def __init__(self, sub_sample=0):
        self.tx = None
        self.y = None
        self.tx_test_labeled = None
        self.y_test_labeled = None
        self.test = None

        self._load_data(sub_sample)

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
                'tx_te': self.tx_test_labeled,
                'y_te': self.y_test_labeled
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

        tx = tx[:, 2:]  # get train data minus the ids and labels
        test = test[:, 2:]  # get test data minus the ids and labels

        # convert class labels from strings to binary (0,1)
        yb = np.ones(len(y))
        yb[np.where(y == 'b')] = 0

        # sub-sample
        if sub_sample:
            yb = yb[::50]
            tx = tx[::50]

        # tx, tx_te, yb, yb_te = self.split_data(tx, yb, 0.9)

        self.tx = tx
        self.y = yb
        # self.tx_test_labeled = tx_te
        # self.y_test_labeled = yb_te
        self.test = test

    def compute_statistics(self):
        """

        :return:
        """
        pass

    @staticmethod
    def split_data(x, y, ratio, seed=1):
        """
        Shuffle the dataset and then split them based on the split ratio.
        The split of the data is always rounded up to the next integer.
        e.g
            number of samples: 5, and ration: 0.5
            split: 5 * 0.5 = 2.5 => 3

        :param x: a numpy array, representing the given features
        :param y: a numpy array, representing the given labels
        :param ratio: float, the ratio of the training data
        :param seed: int, seed number for the shuffling
        :return:
            x_tr: a numpy array representing the given features on the training set
            y_tr: a numpy array representing the labels on the training set
            x_te: a numpy array representing the given features on the testing set
            y_te: a numpy array representing the labels on the testing set

        """
        # set seed
        np.random.seed(seed)
        # generate random indices
        num_rows = len(y)
        # shuffle indexes and then split them
        indices = np.random.permutation(num_rows)
        index_split = int(np.floor(ratio * num_rows))
        index_tr = indices[: index_split]
        index_te = indices[index_split:]
        # create split to data
        x_tr = x[index_tr]
        x_te = x[index_te]
        y_tr = y[index_tr]
        y_te = y[index_te]

        return x_tr, x_te, y_tr, y_te
