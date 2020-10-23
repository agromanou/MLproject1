#!/usr/bin/env python3

"""
Module description
"""
import matplotlib.pyplot as plt


def plot_errors(loss_train, loss_val, jet, mass):
    plt.plot(list(range(len(loss_train))), loss_train, 'g', label='Training loss')
    plt.plot(list(range(len(loss_val))), loss_val, 'b', label='Validation loss')
    plt.title('Training and Validation loss for jet: {jet}'.format(jet=jet))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
