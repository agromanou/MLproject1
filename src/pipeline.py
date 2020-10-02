'''
functions for pipeline
'''

import numpy as np

def transform(tX, col_mean, col_std):
    # Standardize
    tX = (tX - col_mean)/col_std

    # Replaces missings with mean
    inds = np.where(np.isnan(tX))
    tX[inds] = np.take(col_mean, inds[1])

    # Creates dummies for imputed values
    tX_imputed = np.zeros((tX.shape[0],tX.shape[1]))
    array_one = np.ones(tX.shape[1])
    tX_imputed[inds] = np.take(array_one, inds[1])

    # Concatenates imputed dummies with variables
    tX_clean = np.hstack((tX, tX_imputed))
    return tX_clean


def preprocessing(tX, tX_test):
    '''
    Fit and transform both datasets
    '''
    #missing data
    tX = np.where(tX == -999, np.NaN, tX)
    tX_test = np.where(tX_test == -999, np.NaN, tX_test)

    # Our own transformer. fit
    col_mean = np.nanmean(tX, axis=0)
    col_std = np.nanstd(tX, axis=0)

    tX_clean = transform(tX, col_mean, col_std)
    tX_test_clean = transform(tX_test, col_mean, col_std)

    return tX_clean, tX_test_clean
