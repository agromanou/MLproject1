#!/usr/bin/env python3

"""
Module description
"""
import click
import itertools as it
import datetime

from data_loader import DataLoader
from preprocessing import *
from models import LogisticRegression
from visualizations import plot_errors
from utils import build_k_indices
from evaluation import Evaluation


def settings_combinations(search_space):
    all_names = sorted(search_space)
    settings = list(it.product(*(search_space[Name] for Name in all_names)))

    return settings


@click.command()
@click.option('-p1', '--param1', required=False, default=0)
@click.option('-p2', '--param2', required=False, default=0)
def main(param1, param2):

    # Load the data
    data_obj = DataLoader()

    group_labels = list(range(4))
    groups = dict()

    # split data into 4 groups based on their value in the `jet` feature
    for label in group_labels:
        y_, tx_ = get_jet_data_split(data_obj.y, data_obj.tx, label)

        print('\nJet: {}'.format(label))
        print('Original shape: {}'.format(tx_.shape))

        tx_ = remove_empty_features(tx_)  # remove empty features
        tx_ = remove_constant_features(tx_)  # remove constant features

        # treat outliers & missing data
        q1, q3, median = standardization(tx_)
        tx_ = replace_outliers(tx_, q1, q3, median)
        tx_, tx_imputed = treat_missing_data(tx_)

        print('Final shape: {}'.format(tx_.shape))

        groups[str(label)] = {
            'y': y_,
            'tx': tx_,
            'imputed': tx_imputed
        }

    model_parameters = {
        'folds': [5, 10],
        'gamma': [0.0000001, 0.000001, 0.00001, 0.0001],
        'lambda': [0.0000001, 0.000001, 0.00001, 0.0001],
    }

    # get all the possible combinations of settings
    model_settings = settings_combinations(model_parameters)

    print('\nHyper-parameter will run for {} model settings'.format(len(model_settings)))

    best_model = {'f1': 0}
    for model_setting in model_settings:  # loop for mode hyper-parameter tuning

        print('Current setting running: {}'.format(model_setting))

        # Training
        folds = model_setting[0]
        gamma = model_setting[1]
        lambda_ = model_setting[2]
        f1_sum = 0

        # Run for only one jet
        tx = groups['0']['tx']
        y = groups['0']['y']

        # Run cross-validation
        k_indices = build_k_indices(y, folds)
        for k in range(folds):
            # Create indices for the train and validation sets
            val_indice = k_indices[k]
            tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
            tr_indice = tr_indice.reshape(-1)

            x_train = tx[tr_indice]
            x_val = tx[val_indice]
            y_train = y[tr_indice]
            y_val = y[val_indice]

            model = LogisticRegression(x_train, y_train, x_val, y_val)  # instantiate model object
            initial_w = np.zeros(x_train.shape[1])  # initiate the weights

            # Training
            start_time = datetime.datetime.now()
            training_error, validation_error = model.fit(initial_w=initial_w,
                                                         gamma=gamma,
                                                         lambda_=lambda_,
                                                         verbose=False)
            execution_time = (datetime.datetime.now() - start_time).total_seconds()

            print("Gradient Descent: execution time={t:.3f} seconds".format(t=execution_time))

            # Plotting
            plot_errors(loss_train=training_error, loss_val=validation_error)

            # evaluation of validation
            pred = (model.predict(x_val) >= 1/2).astype(int)

            eval_obj = Evaluation(y, pred)
            f1_sum += eval_obj.get_f1()

        avg_f1 = f1_sum / folds

        if best_model['f1'] < avg_f1:
            best_model['f1'] = avg_f1
            best_model['gamma'] = gamma
            best_model['lambda'] = lambda_
            best_model['folds'] = folds

    print('\nBest model:')
    print(best_model)


if __name__ == '__main__':
    main()
