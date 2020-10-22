import numpy as np

from .data_loader import DataLoader
from .preprocessing import *
from .models import *
from .proj1_helpers import create_csv_submission


def best_model_predictions(data_obj, jet):
    y, tx = get_jet_data_split(data_obj.y, data_obj.tx, jet)
    ids_test,tx_test = get_jet_data_split(data_obj.ids_test,data_obj.test, jet)

    file_name ="./../results/gridsearch/results_{0}.csv".format(jet)
    results= np.genfromtxt(file_name, delimiter=",")

    top_row = np.argmax(results[:,6])
    best_params = results[top_row]

    degrees = int(best_params[0])
    features =  int(best_params[1])
    lambda_ = int(best_params[2])
    gamma = int(best_params[3])
    max_iter = int(best_params[4])

    data_cleaner = DataCleaning()
    tx = data_cleaner.fit_transform(tx)
    tx_test = data_cleaner.transform(tx_test)

    feature_generator = FeatureEngineering()
    tx = feature_generator.fit_transform(tx, y,degrees, features)
    tx_test = feature_generator.transform(tx_test)

    initial_w =  np.zeros((tx.shape[1]))
    model = Models()
    w, loss =  model.reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma)
    model.w = w
    pred = model.predict(tx_test)
    return ids_test, pred


def main():
    data_obj = DataLoader()

    ids_test_sub_0, y_pred_0  = best_model_predictions(data_obj, 0)
    ids_test_sub_1, y_pred_1  = best_model_predictions(data_obj, 1)
    ids_test_sub_2, y_pred_2  = best_model_predictions(data_obj, 2)
    ids_test_sub_3, y_pred_3  = best_model_predictions(data_obj, 3)

    ids_all = np.concatenate((ids_test_sub_0,ids_test_sub_1,
                            ids_test_sub_2,ids_test_sub_3), axis = 0)
    preds_all = np.concatenate((y_pred_0,y_pred_1,y_pred_2,y_pred_3),axis = 0)

    preds_all = np.where(preds_all == 0, -1, preds_all)
    OUTPUT_PATH = './../results/predictions/best_model_predictions.csv'
    create_csv_submission(ids_all, preds_all,OUTPUT_PATH)


if __name__ == '__main__':
    main()
