'''
functions for pipeline
'''

import numpy as np


def fix_missings(tX):
    tX = np.where(tX == -999, np.NaN, tX)
    return tX

def remove_empty_features(tX):
    tX = tX[:,~np.all(np.isnan(tX), axis=0)]
    return tX

def remove_constant_features(tX, constant_ind):
    tX = np.delete(tX, constant_ind, axis=1)
    return tX

def replace_outliers(tX, q1,q3,IQR):
    outliers = np.where(tX > q3 + 1.5*IQR)
    tX[outliers] = np.take(q3 + 1.5*IQR, outliers[1])
    outliers = np.where(tX < q1 - 1.5*IQR)
    tX[outliers] = np.take(q1 + 1.5*IQR, outliers[1])
    return tX

def standardize_robust(tX, median, IQR):
    tX = (tX - median)/IQR
    return tX

def treat_missings(tX):
    # Fill na with median (in this case is 0)
    median = 0
    inds = np.where(np.isnan(tX))
    tX[inds] = np.take(median, inds[1])

    # Creates dummies for imputed values
    tX_imputed = np.zeros((tX.shape[0],tX.shape[1]))
    array_one = np.ones(tX.shape[1])
    tX_imputed[inds] = np.take(array_one, inds[1])

    return tX, tX_imputed

def select_top_vars(tX, y, n=10):
    '''
    TODO
    '''
    # logistic regression vars alone
    # order, return top 10
    f1_scores = []
    names = []
    for i in range(tX.shape[0]):
        f1 = logistic_regression(tX,y)
        f1_scores.append(f1)
    return names

def create_squared_features(tX):
    return tX**2

def create_interactions(tX):
    '''
    TODO
    '''
    return tX**2

def create_features(tX,top_vars):
    top_tX = tX[:,top_vars]
    squared =  create_squared_features(top_tX)
    interactions = create_interactions(top_tX)
    extra_features = np.hstack((squared, interactions))
    return extra_features

def preprocessing(tX_train, y_train, tX_test):
    '''
    Fit and transform both datasets
    '''
    #missing data
    tX_train = fix_missings(tX_train)
    tX_test = fix_missings(tX_test)

    #Remove columns with all NaN
    tX_train = remove_empty_features(tX_train)
    tX_test = remove_empty_features(tX_test)

    # Our own transformer. fit
    col_std = np.nanstd(tX_train, axis=0)
    constant_ind = np.where(col_std==0)[0]

    tX_train = remove_constant_features(tX_train, constant_ind)
    tX_test  = remove_constant_features(tX_test constant_ind)

    # Robust standarization & outliers
    q1= np.nanpercentile(tX_train, q = 25, axis=0)
    median =  np.nanpercentile(tX_train, q = 50, axis=0)
    q3 =  np.nanpercentile(tX_train, q = 75, axis=0)
    IQR = q3 - q1

    #Deal with outliers
    tX_train = replace_outliers(tX_train, q1,q3,IQR)
    tX_test = replace_outliers(tX_train, q1,q3,IQR)

    # Robust standarization
    tX_train = standardize_robust(tX_train, median, IQR)
    tX_test = standardize_robust(tX_test, median, IQR)

    # Fill na with median (in this case is 0)
    tX_train, tX_train_imputed  = treat_missings(tX_train)
    tX_test, tX_test_imputed  = treat_missings(tX_test)

    # Concatenates imputed dummies with variables
    top_vars = select_top_vars(tX_train,10)
    extra_features_train = create_features(tX_train,top_vars)
    tX_train = np.hstack((tX_train, extra_features_train, tX_train_imputed))

    extra_features_test = create_features(tX_test,top_vars)
    tX_test = np.hstack((tX_test, extra_features_test, tX_test_imputed))

    return tX_train, tX_test
