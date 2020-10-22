#!/usr/bin/env python3

"""
Module description
"""
import itertools as it

from .data_loader import DataLoader
from .preprocessing import *
from .models import Models
from .evaluation import Evaluation


def settings_combinations(search_space):
    all_names = sorted(search_space)
    settings = list(it.product(*(search_space[Name] for Name in all_names)))
    return settings


def pipeline(x_train, y_train, x_val, y_val, degrees, features, gamma,
             lambda_, epochs, batch_size, verbose):

    print("\t\tData pre-processing & Feature extraction")
    data_cleaner = DataCleaning()
    x_train = data_cleaner.fit_transform(x_train)
    x_val = data_cleaner.transform(x_val)

    feature_generator = FeatureEngineering()
    x_train = feature_generator.fit_transform(x_train, y_train,
                                              degrees, features)
    x_val = feature_generator.transform(x_val)

    model = Models(x_train, y_train, x_val, y_val)  # instantiate model object

    print("\t\tModel Training")

    initial_w = np.zeros(x_train.shape[1])  # initiate the weights
    training_error, validation_error = model.fit(initial_w=initial_w,
                                                 gamma=gamma,
                                                 lambda_=lambda_,
                                                 epochs=epochs,
                                                 batch_size=batch_size,
                                                 verbose=verbose)

    print("\t\tGradient Descent with training error: {train_loss:.4f} \
           and validation error: {val_loss:.4f}".format(
        train_loss=training_error[-1],
        val_loss=validation_error[-1]))

    # evaluation of validation
    pred = model.predict(x_val)
    evaluator = Evaluation(y_val, pred)
    return evaluator.get_f1(), evaluator.get_accuracy()


def cross_validation(tx, y, folds, degrees, features, gamma, lambda_,
                     epochs, batch_size, verbose):
    f1_scores = []
    accuracy_scores = []

    num_rows = len(y)
    interval = int(num_rows / folds)
    indices = np.random.permutation(num_rows)

    for fold in range(int(folds)):
        print('\tFold: {fold}/{folds}'.format(fold=fold+1, folds=folds))
        test_indices = indices[fold * interval: (fold + 1) * interval]
        train_indices = [i for i in indices if i not in test_indices]

        x_train = tx[train_indices]
        y_train = y[train_indices]

        x_val = tx[test_indices]
        y_val = y[test_indices]

        f1, accuracy = pipeline(x_train, y_train, x_val, y_val, degrees,
                                features, gamma, lambda_, epochs, batch_size, verbose)

        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    return np.mean(f1_scores), np.std(f1_scores), \
           np.mean(accuracy_scores), np.std(accuracy_scores)


def model_selection(tx, y, jet, verbose=False):
    model_parameters = {
        'batch_size': [0],
        'degrees_list': [1, 2, 4],
        'epochs': [500],
        'features_list': [4, 6, 8],
        'folds': [5],
        'gamma': [0.000001, 0.00001, 0.0001],
        'lambda': [0.000001, 0.00001, 0.0001]
    }

    # model_parameters = {
    #     'batch_size': [256],
    #     'degrees_list': [2],
    #     'epochs': [500],
    #     'features_list': [4],
    #     'folds': [10],
    #     'gamma': [0.00001],
    #     'lambda': [0.00001]
    # }

    # get all the possible combinations of settings
    model_settings = settings_combinations(model_parameters)

    print('\nHyper-parameter will run for {} model settings'.format(len(model_settings)))

    best_model = {'f1': 0}
    results_list = []

    for idx, model_setting in enumerate(model_settings):  # loop for mode hyper-parameter tuning

        # Training
        batch_size = model_setting[0]
        degrees = model_setting[1]
        epochs = model_setting[2]
        features = model_setting[3]
        folds = model_setting[4]
        gamma = model_setting[5]
        lambda_ = model_setting[6]

        print('\nCurrent setting running ({model_setting}/{total_settings}): K-folds: {folds}, gamma: {gamma}, '
              'lambda: {lambda_}, batch_size: {batch_size}, degrees: {degrees}, features: {features}'.format(
            model_setting=idx + 1, total_settings=len(model_settings), folds=folds, gamma=gamma,
            lambda_=lambda_, batch_size=batch_size, degrees=degrees, features=features))

        f1_mean, f1_std, acc_mean, acc_std = cross_validation(tx, y, folds, degrees, features, gamma, lambda_,
                                                              epochs, batch_size, verbose)

        results = [degrees, features, lambda_, gamma, epochs, folds, f1_mean,
                   f1_std, acc_mean, acc_std]

        results_list.append(results)
        print('Avg F1 score on these settings: {f1:.6f}'.format(f1=f1_mean))

        # keep best model
        if best_model['f1'] < f1_mean:
            best_model['f1'] = f1_mean
            best_model['gamma'] = gamma
            best_model['lambda'] = lambda_
            best_model['folds'] = folds
            best_model['degrees'] = degrees
            best_model['features'] = features
            best_model['epochs'] = epochs

    print('\nBest model for Jet {jet}'.format(jet=jet))
    print(best_model)
    return np.array(results_list)


def main(jet, verbose=False):
    # Load the data
    data_obj = DataLoader()

    jets = list(range(4))
    # Can be run for specific jet
    if jet != -1:
        jets = [jet]

    # split data into 4 groups based on their value in the `jet` feature
    for jet in jets:
        y, tx = get_jet_data_split(data_obj.y, data_obj.tx, jet)

        print('\n' + '-' * 50 + '\n')
        print('RUNNING MODEL SELECTION FOR JET {jet}'.format(jet=jet))

        results = model_selection(tx, y, jet, verbose=verbose)
        file_name = "./../results/gridsearch_results_{0}.csv".format(jet)
        np.savetxt(file_name, results, delimiter=",")


if __name__ == '__main__':
    main(0, False)
