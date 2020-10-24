import numpy as np

from data_loader import *
from preprocessing import *
from implementations import *
from proj1_helpers import *


def best_model_predictions(data_obj, jet, degrees, features):
    """
    This method splits the data based on the jet value
    trains the model and gets the predictions on the test dataset.

    :param data_obj: DataLoader obj
    :param jet: int, the jet value
    :return:
        pred:
        ids:
    """
    # Split data based on jet value for train and val datasets
    y, tx = get_jet_data_split(data_obj.y, data_obj.tx, jet)
    ids_test, tx_test = get_jet_data_split(data_obj.ids_test, data_obj.test, jet)

    data_cleaner = DataCleaning()
    tx = data_cleaner.fit_transform(tx)
    tx_test = data_cleaner.transform(tx_test)

    feature_generator = FeatureEngineering()
    tx = feature_generator.fit_transform(tx, y,degrees, features)
    tx_test = feature_generator.transform(tx_test)

    initial_w =  np.zeros((tx.shape[1]))
    lambda_ = 1e-06
    gamma = 1e-06
    max_iter = 1000

    w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma)
    pred = predict_labels(w, tx_test, True)

    return ids_test, pred


def main():
    """
    The main function that initializes the final training and prediction of the proposed models.
    """
    # Load train and test datasets
    data_obj = DataLoader()

    ids_test_sub_0, y_pred_0 = best_model_predictions(data_obj=data_obj, jet=0, degrees=6, features=2)
    ids_test_sub_1, y_pred_1 = best_model_predictions(data_obj=data_obj, jet=1, degrees=6, features=5)
    ids_test_sub_2, y_pred_2 = best_model_predictions(data_obj=data_obj, jet=2, degrees=6, features=6)
    ids_test_sub_3, y_pred_3 = best_model_predictions(data_obj=data_obj, jet=3, degrees=6, features=4)

    ids_all = np.concatenate((ids_test_sub_0, ids_test_sub_1, ids_test_sub_2, ids_test_sub_3), axis=0)
    preds_all = np.concatenate((y_pred_0, y_pred_1, y_pred_2, y_pred_3), axis=0)

    preds_all = np.where(preds_all == 0, -1, preds_all)
    OUTPUT_PATH = './../results/predictions/best_model_predictions_all.csv'

    create_csv_submission(ids_all, preds_all, OUTPUT_PATH)
    print("Predictions have been created.")


if __name__ == '__main__':
    main()
