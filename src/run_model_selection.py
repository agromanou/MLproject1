#!/usr/bin/env python3

"""
Module description
"""
from data_loader import DataLoader
from preprocessing import *
from evaluation import Evaluation
from proj1_helpers import *
from implementations import *


def pipeline(x_train, y_train, x_val, y_val, degrees, features, gamma,
             lambda_, epochs, verbose):

    data_cleaner = DataCleaning()
    x_train = data_cleaner.fit_transform(x_train)
    x_val = data_cleaner.transform(x_val)

    feature_generator = FeatureEngineering()
    x_train = feature_generator.fit_transform(x_train, y_train,
                                              degrees, features)
    x_val = feature_generator.transform(x_val)

    initial_w = np.zeros(x_train.shape[1])
    w, loss = reg_logistic_regression(y_train,x_train, lambda_ ,initial_w,
                                    epochs, gamma, verbose)

    pred = predict_labels(w,x_val, 1)

    evaluator = Evaluation(y_val, pred)
    return evaluator.get_f1(), evaluator.get_accuracy()


def cross_validation(tx, y, folds, degrees, features, gamma, lambda_,
                     epochs, verbose):
    f1_scores = []
    accuracy_scores = []

    num_rows = len(y)
    interval = int(num_rows / folds)
    indices = np.random.permutation(num_rows)

    for fold in range(int(folds)):
        test_indices = indices[fold * interval: (fold + 1) * interval]
        train_indices = [i for i in indices if i not in test_indices]

        x_train = tx[train_indices]
        y_train = y[train_indices]

        x_val = tx[test_indices]
        y_val = y[test_indices]

        f1, accuracy = pipeline(x_train, y_train, x_val, y_val, degrees,
                                features, gamma, lambda_, epochs, verbose)

        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        print('\tFold: {fold}/{folds}. Results: f1 = {f1}, accuracy = {accuracy}'
        .format(fold=fold+1, folds=folds, f1 = f1, accuracy= accuracy))

    return np.mean(f1_scores), np.std(f1_scores), \
           np.mean(accuracy_scores), np.std(accuracy_scores)


def model_selection(tx, y, jet, verbose=False):

    model_parameters = {
     'degrees_list': list(np.linspace(1,10,10, dtype=int)),
     'epochs': [100, 200, 500, 1000, 2000],
     'features_list': list(np.linspace(1,10,10, dtype=int)),
     'folds': [3,5,10,20],
     'gamma': list(10.**np.arange(-12, 3)),
     'lambda': list(10.**np.arange(-12, 3))
    }

    model_settings = settings_combinations(model_parameters)

    print('\nHyper-parameter will run for {} model settings'
    .format(len(model_settings))) if verbose else None

    best_model = {'f1': 0}
    results_list = []

    for idx, model_setting in enumerate(model_settings):  # loop for mode hyper-parameter tuning

        degrees = model_setting[0]
        epochs = model_setting[1]
        features = model_setting[2]
        folds = model_setting[3]
        gamma = model_setting[4]
        lambda_ = model_setting[5]

        f1_mean, f1_std, acc_mean, acc_std = cross_validation(tx, y, folds, degrees,
                                            features, gamma, lambda_, epochs, verbose)

        print('\nCurrent setting running ({model_setting}/{total_settings}): \
               K-folds: {folds}, gamma: {gamma}, lambda: {lambda_} degrees: \
               {degrees}, features: {features}. \
               Results (f1: {f1}, accuracy: {accuracy})'.format(
            model_setting=idx + 1, total_settings=len(model_settings),
            folds=folds, gamma=gamma,lambda_=lambda_, degrees=degrees,
            features=features, f1=f1_mean, accuracy= acc_mean))

        results = [degrees, features, lambda_, gamma, epochs, folds, f1_mean,
                   f1_std, acc_mean, acc_std]

        results_list.append(results)

    return np.array(results_list)


def main(jet=-1, verbose=False):
    data_obj = DataLoader()

    jets = list(range(4))
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
    main()
