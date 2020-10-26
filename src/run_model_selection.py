#!/usr/bin/env python3

import numpy as np

from data_loader import DataLoader
from preprocessing import FeatureEngineering, DataCleaning, get_jet_data_split
from evaluation import Evaluation
from proj1_helpers import predict_labels, settings_combinations
from implementations import reg_logistic_regression


def pipeline(tx_train, y_train, tx_val, y_val, degrees, gamma,
             lambda_, epochs, verbose):
    """ Run the model training and evaluation on the given parameters """

    # Perform data cleaning (missing values, constant features, outliers, standardization)
    data_cleaner = DataCleaning()
    tx_train = data_cleaner.fit_transform(tx_train)
    tx_val = data_cleaner.transform(tx_val)

    # Perform feature engineering
    feature_generator = FeatureEngineering()
    x_train = feature_generator.fit_transform(tx=tx_train, degree=degrees)
    x_val = feature_generator.transform(tx=tx_val)

    # Initialize values
    initial_w = np.zeros(x_train.shape[1])
    # Train model
    w, _ = reg_logistic_regression(y_train, x_train, lambda_, initial_w,
                                   epochs, gamma, verbose)

    # Perform inference on validation
    pred = predict_labels(weights=w, data=x_val, logistic=True)

    evaluator = Evaluation(y_val, pred)
    return evaluator.get_f1(), evaluator.get_accuracy()


def cross_validation(tx, y, folds, degrees, gamma, lambda_, epochs, verbose):
    """
    It performs k-fold cross validation of the models with the given input hyper-parameters

    :param tx: np.array with the features
    :param y: np.array with the labels
    :param folds: int, the number of folds to perform in k-fold cross validation
    :param degrees: int, the polynomial degree
    :param gamma: float, the learning step
    :param lambda_: float, the lambda penalized term
    :param epochs: int, the number of iterations
    :param verbose: boolean, whether to print detailed log or not
    :return: scores of the performance  of the model
    """

    f1_scores = []
    accuracy_scores = []

    num_rows = len(y)
    interval = int(num_rows / folds)
    np.random.seed(1)
    indices = np.random.permutation(num_rows)

    for fold in range(int(folds)):
        test_indices = indices[fold * interval: (fold + 1) * interval]
        train_indices = [i for i in indices if i not in test_indices]

        # Split train and validations tests
        tx_train = tx[train_indices]
        y_train = y[train_indices]

        tx_val = tx[test_indices]
        y_val = y[test_indices]

        f1, accuracy = pipeline(tx_train, y_train, tx_val, y_val, degrees,
                                gamma, lambda_, epochs, verbose)

        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        print('\tFold: {fold}/{folds}. Results: f1 = {f1}, accuracy = {accuracy}'
              .format(fold=fold + 1, folds=folds, f1=f1, accuracy=accuracy))

    return np.mean(f1_scores), np.std(f1_scores), \
           np.mean(accuracy_scores), np.std(accuracy_scores)


def model_selection(tx, y, verbose=False):

    # Hyper-parameters to test
    model_parameters = {
        'degrees_list': [6, 10],
        'epochs': [1000],
        'folds': [5],
        'gamma': [1e-09, 1e-08, 1e-07, 1e-06],
        'lambda': [1e-09, 1e-08, 1e-07, 1e-06]
    }

    # Calculate the combinations of hyper-parameters values to produce the settings that model selection will run
    model_settings = settings_combinations(model_parameters)

    print('\nHyper-parameter will run for {} model settings' .format(len(model_settings))) if verbose else None

    results_list = []
    for idx, model_setting in enumerate(model_settings):  # loop for mode hyper-parameter tuning

        degrees = model_setting[0]
        epochs = model_setting[1]
        folds = model_setting[2]
        gamma = model_setting[3]
        lambda_ = model_setting[4]

        print('\nCurrent setting running ({model_setting}/{total_settings}): \
               K-folds: {folds}, gamma: {gamma}, lambda: {lambda_} degrees: \
               {degrees}.'.format(
            model_setting=idx + 1, total_settings=len(model_settings),
            folds=folds, gamma=gamma, lambda_=lambda_, degrees=degrees))

        f1_mean, f1_std, acc_mean, acc_std = cross_validation(tx, y, folds, degrees,
                                                              gamma, lambda_, epochs, verbose)

        print('Results (f1: {f1}, accuracy: {accuracy})'.format(f1=f1_mean, accuracy=acc_mean))

        results = [degrees, lambda_, gamma, epochs, folds, f1_mean,
                   f1_std, acc_mean, acc_std]

        results_list.append(results)

    return np.array(results_list)


def main(verbose=False):
    data_obj = DataLoader()

    # split data into 4 groups based on their value in the `jet` feature
    for jet in list(range(4)):
        y, tx = get_jet_data_split(data_obj.y, data_obj.tx, jet)

        print('\n' + '-' * 50 + '\n')
        print('RUNNING MODEL SELECTION FOR JET {jet}'.format(jet=jet))

        # Run model selection
        results = model_selection(tx, y, verbose=verbose)

        # Save results of hyper-parameter tuning
        file_name = "./../results/gridsearch_results_{0}.csv".format(jet)
        np.savetxt(file_name, results, delimiter=",")


if __name__ == '__main__':
    main(verbose=True)
