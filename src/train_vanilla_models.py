# Useful starting lines

from data_loader import DataLoader
from preprocessing import *
from implementations import *
from evaluation import *
from proj1_helpers import predict_labels
from pprint import pprint

import sys
sys.path.append('./../src/')


def run_gradient_descent(tx_train, y_train, tx_val, y_val):
    print('\nTraining with Gradient Descent')
    # Train on all the training dataset and evaluate on validation set
    initial_w = np.zeros((tx_train.shape[1]))
    gamma = 0.05
    max_iter = 1000
    # training
    w, _ = least_squares_GD(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=max_iter,
                            gamma=gamma, verbose=False)

    # predictions
    y_pred = predict_labels(weights=w, data=tx_val, logistic=False)

    evaluation = Evaluation(y_actual=y_val, y_pred=y_pred)
    acc = evaluation.get_accuracy()
    f1 = evaluation.get_f1()
    print('Accuracy: {acc}, F1: {f1}'.format(acc=acc, f1=f1))

    return acc, f1


def run_stochastic_gradient_descent(tx_train, y_train, tx_val, y_val):
    print('\nTraining with Stochastic Gradient Descent')
    initial_w = np.zeros((tx_train.shape[1]))
    gamma = 0.005
    max_iter = 3000

    # training
    w, _ = least_squares_SGD(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=max_iter,
                             gamma=gamma, verbose=False)

    # predictions
    y_pred = predict_labels(weights=w, data=tx_val, logistic=False)

    evaluation = Evaluation(y_actual=y_val, y_pred=y_pred)
    acc = evaluation.get_accuracy()
    f1 = evaluation.get_f1()
    print('Accuracy: {acc}, F1: {f1}'.format(acc=acc, f1=f1))

    return acc, f1


def run_least_squares(tx_train, y_train, tx_val, y_val):
    print('\nTraining with least squares')
    w, _ = least_squares(y=y_train, tx=tx_train)

    # predictions
    y_pred = predict_labels(weights=w, data=tx_val, logistic=False)

    evaluation = Evaluation(y_actual=y_val, y_pred=y_pred)
    acc = evaluation.get_accuracy()
    f1 = evaluation.get_f1()
    print('Accuracy: {acc}, F1: {f1}'.format(acc=acc, f1=f1))

    return acc, f1


def run_ridge_regression(tx_train, y_train, tx_val, y_val):
    print('\nTraining with ridge regression')
    lambda_ = 1e-6

    w, _ = ridge_regression(y=y_train, tx=tx_train, lambda_=lambda_)

    # predictions
    y_pred = predict_labels(weights=w, data=tx_val, logistic=False)

    evaluation = Evaluation(y_actual=y_val, y_pred=y_pred)
    acc = evaluation.get_accuracy()
    f1 = evaluation.get_f1()
    print('Accuracy: {acc}, F1: {f1}'.format(acc=acc, f1=f1))

    return acc, f1


def run_logistic_regression(tx_train, y_train, tx_val, y_val, batch_size):
    s = '\nTraining with logistic regression '
    s += 'with sgd' if batch_size else 'with gd'
    print(s)
    initial_w = np.zeros((tx_train.shape[1]))
    gamma = 1e-6
    max_iter = 3000

    w, _ = logistic_regression(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=max_iter,
                               gamma=gamma, batch_size=batch_size)

    # predictions
    y_pred = predict_labels(weights=w, data=tx_val, logistic=True)

    evaluation = Evaluation(y_actual=y_val, y_pred=y_pred)
    acc = evaluation.get_accuracy()
    f1 = evaluation.get_f1()
    print('Accuracy: {acc}, F1: {f1}'.format(acc=acc, f1=f1))

    return acc, f1


def run_regularized_logistic_regression(tx_train, y_train, tx_val, y_val, batch_size):
    s = '\nTraining with regularized logistic regression '
    s += 'with sgd' if batch_size else 'with gd'
    print(s)
    initial_w = np.zeros((tx_train.shape[1]))
    gamma = 1e-6
    max_iter = 3000
    lambda_ = 0.1

    w, _ = reg_logistic_regression(y=y_train, tx=tx_train, initial_w=initial_w, max_iters=max_iter,
                                   gamma=gamma, batch_size=batch_size, lambda_=lambda_)

    # predictions
    y_pred = predict_labels(weights=w, data=tx_val, logistic=True)

    evaluation = Evaluation(y_actual=y_val, y_pred=y_pred)
    acc = evaluation.get_accuracy()
    f1 = evaluation.get_f1()
    print('Accuracy: {acc}, F1: {f1}'.format(acc=acc, f1=f1))

    return acc, f1


def cross_validation(tx, y, folds=5):
    res = dict()

    num_rows = len(y)
    interval = int(num_rows / folds)
    indices = np.random.permutation(num_rows)

    gd_acc, gd_f1 = list(), list()
    sgd_acc, sgd_f1 = list(), list()
    ls_acc, ls_f1 = list(), list()
    rr_acc, rr_f1 = list(), list()
    lr_gd_acc, lr_gd_f1 = list(), list()
    lr_sgd_acc, lr_sgd_f1 = list(), list()
    rlr_gd_acc, rlr_gd_f1 = list(), list()
    rlr_sgd_acc, rlr_sgd_f1 = list(), list()
    for fold in range(int(folds)):
        print('\n')
        print('-' * 50)
        print('\nFold {fold}/{folds}'.format(fold=fold + 1, folds=folds))
        test_indices = indices[fold * interval: (fold + 1) * interval]
        train_indices = [i for i in indices if i not in test_indices]

        tx_train = tx[train_indices]
        y_train = y[train_indices]

        tx_val = tx[test_indices]
        y_val = y[test_indices]

        # Clean the data (missing values, constant features, outliers, standardization)
        data_cleaner = DataCleaning()
        tx_train = data_cleaner.fit_transform(tx_train)
        tx_val = data_cleaner.transform(tx_val)

        acc, f1 = run_gradient_descent(tx_train, y_train, tx_val, y_val)
        gd_acc.append(acc)
        gd_f1.append(f1)

        acc, f1 = run_stochastic_gradient_descent(tx_train, y_train, tx_val, y_val)
        sgd_acc.append(acc)
        sgd_f1.append(f1)

        acc, f1 = run_least_squares(tx_train, y_train, tx_val, y_val)
        ls_acc.append(acc)
        ls_f1.append(f1)

        acc, f1 = run_ridge_regression(tx_train, y_train, tx_val, y_val)
        rr_acc.append(acc)
        rr_f1.append(f1)

        acc, f1 = run_logistic_regression(tx_train, y_train, tx_val, y_val, batch_size=0)
        lr_gd_acc.append(acc)
        lr_gd_f1.append(f1)

        acc, f1 = run_logistic_regression(tx_train, y_train, tx_val, y_val, batch_size=1)
        lr_sgd_acc.append(acc)
        lr_sgd_f1.append(f1)

        acc, f1 = run_regularized_logistic_regression(tx_train, y_train, tx_val, y_val, batch_size=0)
        rlr_gd_acc.append(acc)
        rlr_gd_f1.append(f1)

        acc, f1 = run_regularized_logistic_regression(tx_train, y_train, tx_val, y_val, batch_size=1)
        rlr_sgd_acc.append(acc)
        rlr_sgd_f1.append(f1)

    res['least_squares_gd'] = {'Acc': sum(gd_acc)/len(gd_acc), 'F1': sum(gd_f1)/len(gd_f1)}
    res['least_squares_sgd'] = {'Acc': sum(sgd_acc)/len(sgd_acc), 'F1': sum(sgd_f1)/len(sgd_f1)}
    res['least_squares'] = {'Acc': sum(ls_acc)/len(ls_acc), 'F1': sum(ls_f1)/len(ls_f1)}
    res['ridge_regression'] = {'Acc': sum(rr_acc)/len(rr_acc), 'F1': sum(rr_f1)/len(rr_f1)}
    res['logistic_regression_gd'] = {'Acc': sum(lr_gd_acc)/len(lr_gd_acc), 'F1': sum(lr_gd_f1)/len(lr_gd_f1)}
    res['logistic_regression_sgd'] = {'Acc': sum(lr_sgd_acc)/len(lr_sgd_acc), 'F1': sum(lr_sgd_f1)/len(lr_sgd_f1)}
    res['reg_logistic_regression_gd'] = {'Acc': sum(rlr_gd_acc)/len(rlr_gd_acc), 'F1': sum(rlr_gd_f1)/len(rlr_gd_f1)}
    res['reg_logistic_regression_sgd'] = {'Acc': sum(rlr_sgd_acc)/len(rlr_sgd_acc), 'F1': sum(rlr_sgd_f1)/len(rlr_sgd_f1)}

    return res


def main_jets():
    # Load data
    data_obj = DataLoader()
    results = dict()

    for jet in range(4):
        results[jet] = dict()

        # Split data based on jet value for train and val datasets
        y_train, tx_train = get_jet_data_split(data_obj.y, data_obj.tx, jet)

        print('\n\nTraining models for jet {jet}'.format(jet=jet))
        res = cross_validation(tx=tx_train, y=y_train)
        results[jet] = res

    print('RESULTS:')
    pprint(results)


def main():
    # Load data
    data_obj = DataLoader()

    print('\n\nTraining models')
    res = cross_validation(tx=data_obj.tx, y=data_obj.y)

    print('RESULTS:')
    pprint(res)


if __name__ == '__main__':
    main()
