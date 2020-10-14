#!/usr/bin/env python3

"""
Module description
"""
import click
import itertools as it
import datetime

from src.data_loader import DataLoader
from src.preprocessing import *
from src.models import *
from src.visualizations import plot_errors


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
        y_, tx_, ids_ = get_jet_data_split(data_obj.y, data_obj.tx, data_obj.ids, label)

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
            'ids': ids_,
            'imputed': tx_imputed
        }

    print(groups.keys())

    model_parameters = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'gamma': [0.001, 0.1, 0.5, 1, 2, 10],
        'l1_ratio': [0.25, 0.5, 0.75],
        'model': [Model]
    }

    # get all the possible combinations of settings
    model_settings = settings_combinations(model_parameters)

    print('\nHyper-parameter will run for {} model settings'.format(len(model_settings)))

    for model_setting in model_settings:  # loop for mode hyper-parameter tuning

        print('Current setting running: {}'.format(model_setting))

        # Training
        gamma = model_setting[0]
        penalty = model_setting[1]
        model_class = model_setting[2]
        l1_ratio = model_setting[3]

        tx = groups['0']['tx']
        y = groups['0']['y']

        model = model_class(tx, y)  # instantiate model object
        initial_w = np.zeros(tx.shape[1])  # initiate the weights

        # Training
        start_time = datetime.datetime.now()
        training_error, validation_error = model.fit(gamma=gamma, initial_w=initial_w)
        execution_time = (datetime.datetime.now() - start_time).total_seconds()

        print("Gradient Descent: execution time={t:.3f} seconds".format(t=execution_time))

        # Plotting
        plot_errors(loss_train=training_error, loss_val=validation_error)

        break

    # Evaluation on unseen data
    #test_labelled_error = model.score(data_obj.tx_te, data_obj.y_te)


if __name__ == '__main__':
    main()

