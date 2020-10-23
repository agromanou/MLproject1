
from data_loader import DataLoader
from preprocessing import *
from models import *
from implementations import *
from proj1_helpers import create_csv_submission


def best_model_predictions(data_obj, jet, mass):
    """
    This method splits the data based on the jet value and the mass existence (8 partitions),
    trains the model and gets the predictions on the test dataset.

    :param data_obj: DataLoader obj
    :param jet: int, the jet value
    :return:
        pred:
        ids:
    """
    print('\n\nTraining model for jet {jet} with mass {mass}'.format(jet=jet, mass=mass))

    # Split data based on jet value for train and val datasets
    y, tx = get_jet_data_split(data_obj.y, data_obj.tx, jet)
    ids_test, tx_test = get_jet_data_split(data_obj.ids_test, data_obj.test, jet)

    if mass == 0 or mass == 1:
        # Split data based on mass existence for train and val datasets
        tx, y = get_mass_data_split(y, tx, mass)
        tx_test, ids_test = get_mass_data_split(ids_test, tx_test, mass)

    print('Train data size: {train}  |  Test data size: {test}'.format(train=len(tx), test=len(tx_test)))

    # Read hyper-parameter tuning results for a specific jet with specific mass
    file_name_with_tuning_results = "./../results/gridsearch_results_jet{jet}_mass{mass}.csv".format(jet=jet,
                                                                                                     mass=mass)
    hyper_params_results = np.genfromtxt(file_name_with_tuning_results, delimiter=",", skip_header=1)

    # Select best setting based on f1 score (6th column in the file)
    best_hyper_params_indx = np.argmax(hyper_params_results[:, 6])
    best_hyper_params = hyper_params_results[best_hyper_params_indx]

    # Train and test the model
    preds = get_predictions(best_hyper_params, tx, tx_test, y)

    return preds, ids_test


def get_predictions(params, tx, tx_test, y):
    """
    It performs the training on the full training dataset with the provided hyper-parameters,
    and applies the trained model to the test dataset to get predictions.

    :param params: list with the hyper-parameter setting
    :param tx: np.array of the train dataset
    :param tx_test: np.array of the test dataset
    :param y: np.array of the labels of the train dataset
    :return:
        pred: np.array with the predicted labels
    """
    degrees = int(params[0])
    features = int(params[1])
    lambda_ = params[2]
    gamma = params[3]
    max_iter = int(params[4])

    print('with hyper-parameters: degrees: {degrees}, features: {features}, lambda_: {lambda_}, '
          'gamma: {gamma}, max_iter: {max_iter}'.format(degrees=degrees, features=features, lambda_=lambda_,
                                                        gamma=gamma, max_iter=max_iter))

    print('Feature engineering')
    # Clean the data (missing values, constant features, outliers, standardization)
    data_cleaner = DataCleaning()
    tx = data_cleaner.fit_transform(tx)
    tx_test = data_cleaner.transform(tx_test)

    print('Training')
    # Create new features (polynomial expansion, feature interaction)
    feature_generator = FeatureEngineering()
    tx = feature_generator.fit_transform(tx, y, degrees, features)
    tx_test = feature_generator.transform(tx_test)

    # Train on all the training dataset and predict on test set
    initial_w = np.zeros((tx.shape[1]))
    model = Models()
    w, loss = model.reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma)
    model.w = w
    pred = model.predict(tx_test)

    return pred


def main(with_mass=True):
    """
    The main function that initializes the final training and prediction of the proposed models.
    """
    # Load train and test datasets
    data_obj = DataLoader()

    ids = list()
    preds = list()

    for jet in range(4):  # for each jet
        if with_mass:
            for mass in range(2):  # for each mass

                # For each jet value train the respective model and get predictions
                y_pred, ids_test = best_model_predictions(data_obj=data_obj, jet=jet, mass=mass)
                ids.append(ids_test)
                preds.append(y_pred)
        else:
            # For each jet value train the respective model and get predictions
            y_pred, ids_test = best_model_predictions(data_obj=data_obj, jet=jet, mass=None)
            ids.append(ids_test)
            preds.append(y_pred)

    # Concatenate all predictions along with their ids
    ids_all = np.concatenate(ids, axis=0)
    preds_all = np.concatenate(preds, axis=0)

    # Change class 0 to -1 label
    preds_all = np.where(preds_all == 0, -1, preds_all)

    # Write predictions to file
    OUTPUT_PATH = './../results/predictions/best_model_predictions.csv'
    create_csv_submission(ids_all, preds_all, OUTPUT_PATH)
    print("Predictions have been created.")


if __name__ == '__main__':
    main(with_mass=True)
