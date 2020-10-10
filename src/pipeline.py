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

    #Remove columns with all NaN
    tX_clean = tX_clean[:,~np.all(np.isnan(tX_clean), axis=0)]
    return tX_clean

def data_normalization(tX0_clean):
    for i in range(tX0_clean.shape[1])[:2]:
        col = tX0_clean[:, i]
        col_mean = np.mean(col)
        col_std = np.std(col)

        upper_lim = col_mean + 2 * col_std
        lower_lim = col_mean - 2 * col_std

        big_outliers = np.where(col > upper_lim)
        small_outliers = np.where(col < lower_lim)

        col[big_outliers] = upper_lim
        col[small_outliers] = lower_lim
    return tX0_clean

def preprocessing(tX, tX_test):
    '''
    Fit and transform both datasets
    '''
    #missing data
    tX = np.where(tX == -999, np.NaN, tX)
    tX_test = np.where(tX_test == -999, np.NaN, tX_test)

    # Our own transformer. fit
    col_mean = np.nanmean(tX, axis=1)
    col_std = np.nanstd(tX, axis=1)

    constant_ind = np.where(col_std==0)[0]
    tX = np.delete(tX, constant_ind, axis=1)
    tX_test  = np.delete(tX_test, constant_ind, axis=1)
    col_std = np.delete(col_std, constant_ind, axis=0)
    col_mean = np.delete(col_mean, constant_ind, axis=0)

    tX_clean = transform(tX, col_mean, col_std)
    tX_test_clean = transform(tX_test, col_mean, col_std)

    tX_clean = data_normalization(tX_clean)
    tX_test_clean = data_normalization(tX_test_clean)
    return tX_clean, tX_test_clean
