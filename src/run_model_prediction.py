import numpy as np

from data_loader import DataLoader
from preprocessing import *
from implementations import *


def best_model_predictions(data_obj, jet):
    y, tx = get_jet_data_split(data_obj.y, data_obj.tx, jet)
    ids_test,tx_test = get_jet_data_split(data_obj.ids_test,data_obj.test, jet)

    tx_no_mass, y_no_mass, tx_mass, y_mass = split_jet_by_mass(y, tx)
    tx_test_no_mass, ids_test_no_mass, tx_test_mass, ids_test_mass = split_jet_by_mass(ids_test, tx_test)

    file_name_no_mass ="./../results/gridsearch_results_no_mass_{0}.csv".format(jet)
    results_no_mass= np.genfromtxt(file_name_no_mass, delimiter=",", skip_header=1)

    top_row_no_mass = np.argmax(results_no_mass[:,6])
    best_params_no_mass = results_no_mass[top_row_no_mass]

    pred_no_mass = get_predictions(best_params_no_mass, tx_no_mass, tx_test_no_mass, y_no_mass)

    file_name_with_mass = "./../results/gridsearch_results_with_mass_{0}.csv".format(jet)
    results_with_mass = np.genfromtxt(file_name_with_mass, delimiter=",", skip_header=1)

    top_row_with_mass = np.argmax(results_with_mass[:, 6])
    best_params_with_mass = results_with_mass[top_row_with_mass]

    pred_with_mass = get_predictions(best_params_with_mass, tx_mass, tx_test_mass, y_mass)

    return pred_no_mass, ids_test_no_mass, pred_with_mass, ids_test_mass


def get_predictions(best_params, tx, tx_test, y):
    degrees = int(best_params[0])
    features = int(best_params[1])
    lambda_ = best_params[2]
    gamma = best_params[3]
    max_iter = int(best_params[4])

    data_cleaner = DataCleaning()

    tx = data_cleaner.fit_transform(tx)
    tx_test = data_cleaner.transform(tx_test)

    feature_generator = FeatureEngineering()

    tx = feature_generator.fit_transform(tx, y, degrees, features)
    tx_test = feature_generator.transform(tx_test)

    initial_w = np.zeros((tx.shape[1]))
    model = Models()

    w, loss = model.reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma)
    model.w = w
    pred = model.predict(tx_test)

    return pred


def main():
    data_obj = DataLoader()

    # ids_test_sub_0, y_pred_0  = best_model_predictions(data_obj, 0)
    # ids_test_sub_1, y_pred_1  = best_model_predictions(data_obj, 1)
    # ids_test_sub_2, y_pred_2  = best_model_predictions(data_obj, 2)
    # ids_test_sub_3, y_pred_3  = best_model_predictions(data_obj, 3)

    y_pred_0_no_mass, ids_test_sub_0_no_mass, y_pred_0_with_mass, ids_test_sub_0_with_mass = best_model_predictions(data_obj, 0)
    y_pred_1_no_mass, ids_test_sub_1_no_mass, y_pred_1_with_mass, ids_test_sub_1_with_mass = best_model_predictions(
        data_obj, 1)
    y_pred_2_no_mass, ids_test_sub_2_no_mass, y_pred_2_with_mass, ids_test_sub_2_with_mass = best_model_predictions(
        data_obj, 2)
    y_pred_3_no_mass, ids_test_sub_3_no_mass, y_pred_3_with_mass, ids_test_sub_3_with_mass = best_model_predictions(
        data_obj, 3)

    ids_all = np.concatenate((ids_test_sub_0_no_mass, ids_test_sub_1_no_mass, ids_test_sub_2_no_mass, ids_test_sub_3_no_mass, ids_test_sub_0_with_mass, ids_test_sub_1_with_mass, ids_test_sub_2_with_mass, ids_test_sub_3_with_mass), axis = 0)
    preds_all = np.concatenate((y_pred_0_no_mass ,y_pred_1_no_mass, y_pred_2_no_mass, y_pred_3_no_mass, y_pred_0_with_mass ,y_pred_1_with_mass, y_pred_2_with_mass, y_pred_3_with_mass),axis = 0)

    preds_all = np.where(preds_all == 0, -1, preds_all)
    OUTPUT_PATH = './../results/predictions/best_model_predictions.csv'
    create_csv_submission(ids_all, preds_all,OUTPUT_PATH)


if __name__ == '__main__':
    main()
