'''
functions for pipeline
'''
from imple2 import *
from utils import *
from proj1_helpers import *


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import numpy as np

DATA_TRAIN_PATH = "./../data/raw/train.csv"
DATA_TEST_PATH = "./../data/raw/test.csv"

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
    median = np.zeros(tX.shape[1])
    inds = np.where(np.isnan(tX))
    tX[inds] = np.take(median, inds[1])

    # Creates dummies for imputed values
    tX_imputed = np.zeros((tX.shape[0],tX.shape[1]))
    array_one = np.ones(tX.shape[1])
    tX_imputed[inds] = np.take(array_one, inds[1])

    return tX, tX_imputed

def select_top_vars(tX, y, n=5):
    initial_w =  np.zeros((tX.shape[1]))
    w, loss = reg_logistic_regression(y, tX, 0.01, initial_w, 5000, 1e-6)
    top_n_features = np.argsort(-abs(w))[:n]
    return top_n_features


def create_poly_features(tX, degree):
    tX_poly = build_poly(tX, 3)
    tX_poly = np.delete(tX_poly, 0, 1)
    return tX_poly

def create_interactions(X, columns):
    for col1 in columns:
        for col2 in columns:
            if col1 > col2:
                col3 = np.multiply(X[:, col1], X[:, col2])
                X = np.c_[X, col3]
    return X

def create_features_train(tX, y, degree, num_top_vars):
    tX_poly =  create_poly_features(tX, degree)
    top_tX = select_top_vars(tX, y, num_top_vars)
    interactions = create_interactions(tX_poly, top_tX)
    return interactions, top_tX

def create_features_test(tX, degree, top_tX):
    tX_poly =  create_poly_features(tX, degree)
    interactions = create_interactions(tX_poly, top_tX)
    return interactions

def preprocess(tX_train, tX_test):
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
    tX_test  = remove_constant_features(tX_test, constant_ind)

    # Robust standarization & outliers
    q1= np.nanpercentile(tX_train, q = 25, axis=0)
    median =  np.nanpercentile(tX_train, q = 50, axis=0)
    q3 =  np.nanpercentile(tX_train, q = 75, axis=0)
    IQR = q3 - q1

    #Deal with outliers
    tX_train = replace_outliers(tX_train, q1,q3,IQR)
    tX_test = replace_outliers(tX_test, q1,q3,IQR)

    # Robust standarization
    tX_train = standardize_robust(tX_train, median, IQR)
    tX_test = standardize_robust(tX_test, median, IQR)

    # Fill na with median (in this case is 0)
    tX_train, tX_train_imputed  = treat_missings(tX_train)
    tX_test, tX_test_imputed  = treat_missings(tX_test)

    return tX_train, tX_test, tX_train_imputed, tX_test_imputed





def tune_hyperparameters(X_train, y_train):
    degrees_list = [1,2,3,4,5,6,7,8]
    features_list = [3,4,5,6,7,8,9,10]
    c_list =  [0.001, 0.1, 0.5, 1, 2, 10]
    gamma_list = [0.001, 0.1, 0.5, 1, 2, 10]
    results_list = []

    for degree in degrees_list:
        for features in features_list:
            for c in c_list:
                for gamma in gamma_list:
                    f1_mean, f1_std, acc_mean, acc_std = \
                            cross_validation(X_train, y_train degree, features, c, gamma)
                    results = [degree, features, c, gamma,f1_mean,
                                        f1_std, acc_mean, acc_std]
                    results_list.append(results)

    return np.array(results_list)

def train_model_gs(tX_sub, y_sub, jet_number=0):
    X_test, X_train, y_test, y_train = split_data(tX, y, 0.1, myseed=1)

    results_list = tune_hyperparameters(X_train, y_train)

    # Save results list
    # TO CSV

    ## CHOSE BEST HYPERPARAMETERS
    top_row = np.argmax(results_gs[:,4])
    best_params = results_gs[top_row]

    ## RUN WITH BEST HYPERPARAMETRS
    degree = best_params[0]
    num_top_vars =  best_params[1]
    lambda_ = best_params[2]
    gamma = best_params[3]

    w = run_fit(X_train, y_test, y_train , degree,
                    num_top_vars, lambda_, gamma)

    ## MAKE PREDICTIONS
    return w,  num_top_vars, degree



def visualize_generalization():
    top_row = np.argmax(results_gs[:,4])
    best_params = results_gs[top_row]

    ## RUN WITH BEST HYPERPARAMETRS
    degree = best_params[0]
    num_top_vars =  best_params[1]
    lambda_ = best_params[2]
    gamma = best_params[3]


def run_single_group_gs(tX, y, tX_test, ids_test, group, degree):
    #TRAIN
    inds = np.where(tX[:,22]==group)
    tX_sub = tX[inds]
    y_sub = y[inds]

    w,  top_tX, degree = train_model_gs(tX_sub, y_sub)

    # TEST
    inds = np.where(tX_test[:,22]==group)
    tX_test_sub = tX_test[inds]
    ids_test_sub = ids_test[inds]

    _, tX_test_sub, _, _ = preprocess(tX_sub, tX_test_sub)
    tX_test_sub = create_features_test(tX_test_sub, degree, top_tX)
    y_pred = predict_labels(w, tX_test_sub,1)

    return ids_test_sub, y_pred


def cross_validation(tX_sub, y_sub, degree, num_top_vars, lambda_, gamma, folds=5):
    f1_scores = []
    accuracy_scores = []
    for i in range(folds):
        X_test,X_train, y_test, y_train = split_data_fold(tX_sub, y_sub, myseed=1)
        # GET DATA FROM FOLD
        y_pred, _ =  run_pipeline(X_test, X_train, y_test, y_train , degree,
                                num_top_vars, lambda_, gamma)
        f1, accuracy = calculate_scores(y_test, y_pred)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    return np.mean(f1_score), np.std(f1_score), np.mean(accuracy), np.std(accuracy)


def calculate_scores(y_test, y_pred):
    #George
    return 0

def run_fit(X_train, y_train , degree,
                num_top_vars, lambda_, gamma, max_iters=1):

    tX_train = fix_missings(tX_train)
    tX_train = remove_empty_features(tX_train)

    # Our own transformer. fit
    col_std = np.nanstd(tX_train, axis=0)
    constant_ind = np.where(col_std==0)[0]

    tX_train = remove_constant_features(tX_train, constant_ind)

    # Robust standarization & outliers
    q1= np.nanpercentile(tX_train, q = 25, axis=0)
    median =  np.nanpercentile(tX_train, q = 50, axis=0)
    q3 =  np.nanpercentile(tX_train, q = 75, axis=0)
    IQR = q3 - q1

    #Deal with outliers
    tX_train = replace_outliers(tX_train, q1,q3,IQR)

    # Robust standarization
    tX_train = standardize_robust(tX_train, median, IQR)

    # Fill na with median (in this case is 0)
    tX_train, tX_train_imputed  = treat_missings(tX_train)

    #Features
    X_train =  create_features_test(X_train, degree, num_top_vars)

    # Modelling
    initial_w = np.zeros((tX.shape[1]))
    w, loss = reg_logistic_regression(y_train, X_train, lambda_,
                                    initial_w, max_iters, gamma)

    return w


def run_pipeline(X_test, X_train, y_test, y_train , degree,
                num_top_vars, lambda_, gamma, max_iters=1):
    #Cleaning
    X_train, X_test, _, _ = preprocess(X_train, X_test)

    #Features
    X_train , top_tX =  create_features_train(X_train, y_train,
                                            degree, num_top_vars)
    X_test = create_features_test(X_test, degree, top_tX)

    # Modelling
    initial_w = np.zeros((tX.shape[1]))
    w, loss = reg_logistic_regression(y_train, tX_train, lambda_,
                                    initial_w, max_iters, gamma)
    y_pred = predict_labels(w, X_test)

    return y_pred

def train_model(tX_sub, y_sub, degree,num_top_vars):
    '''
    TODO
    '''
    tX_test,tX_train, y_test, y_train = split_data(tX_sub, y_sub, 0.1, myseed=1)
    X_train, X_test, _, _ = preprocess(tX_train, tX_test)
    X_train , top_tX =  create_features_train(X_train, y_train, degree, num_top_vars)
    # hyperparameter search
    # lambda, gamma


    # TODO: use our methods
    # Cross validation
    clf = LogisticRegression(max_iter = 1500)
    clf.fit(X_train, y_train)

    X_test = create_features_test(X_test, degree, top_tX)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    w = clf.coef_[0]
    return w, top_tX


def run_single_group(tX, y, tX_test, ids_test, group, degree):
    #TRAIN
    inds = np.where(tX[:,22]==group)
    tX_sub = tX[inds]
    y_sub = y[inds]

    w,  top_tX = train_model(tX_sub, y_sub, degree)

    # TEST
    inds = np.where(tX_test[:,22]==group)
    tX_test_sub = tX_test[inds]
    ids_test_sub = ids_test[inds]

    _, tX_test_sub, _, _ = preprocess(tX_sub, tX_test_sub)
    tX_test_sub = create_features_test(tX_test_sub, degree, top_tX)
    y_pred = predict_labels(w, tX_test_sub,1)

    return ids_test_sub, y_pred


def run_all_groups():
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    ids_test_sub_0, y_pred_0  = run_single_group(tX, y, tX_test, ids_test, 0,3) #3
    ids_test_sub_1, y_pred_1  = run_single_group(tX, y, tX_test, ids_test, 1,4) #4
    ids_test_sub_2, y_pred_2  = run_single_group(tX, y, tX_test, ids_test, 2,3) #3
    ids_test_sub_3, y_pred_3  = run_single_group(tX, y, tX_test, ids_test, 3,5) #5

    ids_all = np.concatenate((ids_test_sub_0,ids_test_sub_1,
                            ids_test_sub_2,ids_test_sub_3), axis = 0)
    preds_all = np.concatenate((y_pred_0,y_pred_1,y_pred_2,y_pred_3),axis = 0)

    return ids_all, preds_all
