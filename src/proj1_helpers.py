# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import itertools as it


def settings_combinations(search_space):
    """
    It creates all the combinations of values between the elements of input lists.
    e.g.
        Input:
            {'paramA': ['value1', 'value2'],
             'paramB': ['value3', 'value4']}
        Output:
            [['value1', 'value3'],
            ['value1', 'value4'],
            ['value2', 'value3'],
            ['value2', 'value4']]

    :param search_space: dict of lists, with the possible values for each hyper-parameter.
    :return:
        settings: list of lists with the different combinations of settings.
    """
    params = sorted(search_space)
    settings = list(it.product(*(search_space[param] for param in params)))
    return settings


def sigmoid(x):
    """Apply sigmoid function on x"""
    return 1.0 / (1 + np.exp(-x))


def predict_labels(weights, data, logistic):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = sigmoid(np.dot(data, weights)) if logistic else np.dot(data, weights)

    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True, seed = 1):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    np.random.seed(seed)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
