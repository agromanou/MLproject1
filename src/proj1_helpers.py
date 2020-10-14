# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def predict_labels(weights, data, logistic):
    """Generates class predictions given weights, and a test data matrix"""
    if logistic:
        threshold = 0.5
    else:
        threshold = 0

    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1

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
