#!/usr/bin/env python3

"""
Module description
"""
import click
import itertools as it
import datetime

from data_loader import DataLoader
from preprocessing import *
from implementations import Models
from visualizations import plot_errors
from utils import build_k_indices
from evaluation import Evaluation


def settings_combinations(search_space):
    all_names = sorted(search_space)
    settings = list(it.product(*(search_space[Name] for Name in all_names)))

    return settings



def model_selection(tx, y, jet, verbose=False):
    model_parameters = {
        'degrees_list': [1, 2, 3, 4, 5],
        'epochs': [500, 1000, 10000],
        'features_list': [3, 4, 5, 6, 7],
        'folds': [5, 10],
        'gamma': [0.0000001, 0.000001, 0.00001, 0.0001],
        'lambda': [0.0000001, 0.000001, 0.00001, 0.0001]
    }

    model_parameters = {
        'degrees_list': [2],
        'epochs': [500],
        'features_list': [3],
        'folds': [3],
        'gamma': [0.0000001],
        'lambda': [0.0000001]
    }
    # get all the possible combinations of settings
    model_settings = settings_combinations(model_parameters)

    print('\nHyper-parameter will run for {} model settings'.format(len(model_settings))) if verbose else None

    best_model = {'f1': 0}
    for model_setting in model_settings:  # loop for mode hyper-parameter tuning

        # Training
        degrees = model_setting[0]
        epochs = model_setting[1]
        features = model_setting[2]
        folds = model_setting[3]
        gamma = model_setting[4]
        lambda_ = model_setting[5]

        print('\nCurrent setting running: K-folds: {folds}, gamma: {gamma}, lambda: {lambda_}'.format(
            folds=folds, gamma=gamma, lambda_=lambda_)) if verbose else None

        # Run cross-validation
        k_indices = build_k_indices(y, folds)
        f1_sum = 0
        for k in range(folds):

            # Create indices for the train and validation sets
            val_indice = k_indices[k]
            tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
            tr_indice = tr_indice.reshape(-1)

            x_train = tx[tr_indice]
            x_val = tx[val_indice]
            y_train = y[tr_indice]
            y_val = y[val_indice]

            # data pre-processing
            x_train, x_val = preprocess(x_train, x_val)

            # feature creation
            feature_generator = FeatureEngineering()
            x_train = feature_generator.fit_transform(x_train, y_train, degrees, features)
            x_val = feature_generator.transform(x_val)

            # Training
            model = Models(x_train, y_train, x_val, y_val)  # instantiate model object
            initial_w = np.zeros(x_train.shape[1])  # initiate the weights

            start_time = datetime.datetime.now()
            training_error, validation_error = model.fit(initial_w=initial_w,
                                                         gamma=gamma,
                                                         lambda_=lambda_,
                                                         epochs=epochs,
                                                         verbose=False)
            execution_time = (datetime.datetime.now() - start_time).total_seconds()

            print("\tGradient Descent for {k}/{folds} folds: execution time={t:.3f} seconds "
                  "with training error: {train_loss:.4f} and validation error: {val_loss:.4f}".format(
                k=k+1, folds=folds, t=execution_time, train_loss=training_error[-1],
                val_loss=validation_error[-1])) if verbose else None

            # Plotting
            plot_errors(loss_train=training_error, loss_val=validation_error) if verbose else None

            # evaluation of validation
            pred = (model.predict(x_val) >= 1/2).astype(int)

            eval_obj = Evaluation(y, pred)
            f1_sum += eval_obj.get_f1()

        avg_f1 = f1_sum / folds

        print('Avg F1 score on these settings: {f1:.4f}'.format(f1=avg_f1)) if verbose else None

        # keep best model
        if best_model['f1'] < avg_f1:
            best_model['f1'] = avg_f1
            best_model['gamma'] = gamma
            best_model['lambda'] = lambda_
            best_model['folds'] = folds

    print('\nBest model for Jet {jet}'.format(jet=jet))
    print(best_model)


@click.command()
@click.option('-j', '--jet', required=False, default=-1)
@click.option('-v', '--verbose', required=False, default=True)
def main(jet, verbose):

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

        model_selection(tx, y, jet, verbose=verbose)


if __name__ == '__main__':
    main()
