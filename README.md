# Discovering the Higgs boson

This project is the first group assignment for the Machine Learning course (CS-443) at EPFL. 
It implements a classification machine learning model from scratch 
for the purpose to find the Higgs boson using original data from CERN (Kaggle challenge).

The final model achieved a categorical accuracy of **0.809** and f1 score of **0.730**.

* [Getting started](#getting-started)
    * [Data](#data)
    * [Dependencies](#dependencies)
    * [Report](#report)
* [Project Architecture](#project-architecture)
* [Running model selection](#running-model-selection)
* [Running final model](#running-final-model)


## Getting started
#### Data
The raw data can be downloaded form the webpage of the AIcrowd challenge: \
https://www.aicrowd.com/challenges/epfl-machine-learning-higgs. \
The data should be located in the `data/` directory in csv format.

The documentation of the provided dataset can be found here: \
https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf.


#### Report
Our paper regarding the methodology and the experiments of the proposed model 
is located under the `report/` directory in LaTeX and pdf format. 

#### Dependencies
Project dependencies are located in the `requirements.txt` file. \
To install them you should run:
```bash
pip install -r requirements.txt
```

## Project Architecture
The source code of this project is located under the `src/` directory. 

* **data_loader.py**: Class `DataLoader()` responsible for data loading and splitting. 
* **preprocessing.py**:  Classes `DataCleaning()` and `FeatureEngineering` responsible for missing values imputation, 
treatment of outliers, standardization, normalization, polynomial expansion and feature interaction.
* **implementations.py**: Functions responsible for model training and testing. 
* **evaluation.py**: Class `Evaluation()` responsible for the computation of classification evaluation metrics. 
* **visualization.py**: Functions responsible for data visualization.
* **run_model_selection.py**: `main()` function that runs hyper-parameter tuning and cross-validation, 
storing the performance of each tested model. 
* **run.py**: `main()` function that read the output of `run_model_selection.py`, selects the hyper-parameters
of the model with the best performance, trains the model on all training data and produces predictions on the test dataset.



## Running model selection
To run the model selection process, please run the following command:
```bash
python run_model_selection.py
```

## Running final model
To train the final model and test it in the testing data, please run the following command:
```bash
python run.py
```




