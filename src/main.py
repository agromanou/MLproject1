#!/usr/bin/env python3

"""
Module description
"""
import click
import itertools as it
import datetime

from src.data_loader import DataLoader
from src.preprocessing import BasicPreprocessor
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
    tx = data_obj.tx
    y = data_obj.y

    # set of hyper-params
    preprocessing_settings = {
        'setting_1': [BasicPreprocessor.fill_nas]
    }

    model_parameters = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.1, 0.5, 1, 2, 10],
        'l1_ratio': [0.25, 0.5, 0.75],
        'model': [Model]
    }

    model_settings = settings_combinations(model_parameters)

    for prepro_label, prepro_setting in preprocessing_settings.items():  # loop for a parameter

        # Run data pre-processing
        for preproccesing in prepro_setting:
            tx = preproccesing(tx)

        print(y[:10])

        for model_setting in model_settings:  # loop for mode hyper-parameter tuning

            print(model_setting)

            # Training
            c = model_setting[0]
            penalty = model_setting[1]
            model_class = model_setting[2]
            l1_ratio = model_setting[3]

            model = model_class(tx, y)

            initial_w = np.zeros(tx.shape[1])

            # Training
            start_time = datetime.datetime.now()
            training_error, validation_error = model.fit(gamma=0.5, initial_w=initial_w)
            execution_time = (datetime.datetime.now() - start_time).total_seconds()

            print("Gradient Descent: execution time={t:.3f} seconds".format(t=execution_time))

            # Plotting
            plot_errors(loss_train=training_error, loss_val=validation_error)

            break
        break

    # Evaluation on unseen data
    # test_labelled_error = model.score(tx, y)


if __name__ == '__main__':
    main()

