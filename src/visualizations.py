#!/usr/bin/env python3

import matplotlib.pyplot as plt


def plot_errors(loss_train, loss_val, jet):
    """
    Plots train-validation losses per epoch.

    :param loss_train: np.array of the losses on each epoch for training data
    :param loss_val: np.array of the losses on each epoch for validation data
    :param jet: int, the jet value
    """
    plt.plot(list(range(len(loss_train))), loss_train, 'g', label='Training loss')
    plt.plot(list(range(len(loss_val))), loss_val, 'b', label='Validation loss')
    plt.title('Training and Validation loss for jet: {jet}'.format(jet=jet))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
