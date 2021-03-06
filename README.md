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
* [Running the code](#running-the-code)
    * [Running vanilla models](#running-vanilla-models)
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
The source code of this project is structured in the following manner. 

```
project
│  README.md
│  requirements.txt
│
├─docs/                        # documentation of the problem
│
├─data/                        # the data directory
│  
├─notebooks/                   # experimentation and exploration notebooks
│ 
├─results/
│    predictions/              # directory to store predictions
│ 
└─src/
     data_loader.py            # Class `DataLoader` responsible for data loading and splitting. 
     preprocessing.py          # Classes `DataCleaning` and `FeatureEngineering` responsible for missing values imputation, 
                                 treatment of outliers, standardization, normalization and polynomial expansion.
     implementations.py        # Functions responsible for model training and testing.
     costs.py                  # Functions responsible for loss functions and gradient computations.
     evaluation.py             # Class `Evaluation` responsible for the computation of classification evaluation metrics.
     visualization.py          # Functions responsible for data visualization.
     run_vanilla_models.py     # `main()` function that tests the performance of vanilla models with cross-validation 
                                 without any feature engineering.
     run_model_selection.py    # `main()` function that runs hyper-parameter tuning and cross-validation, 
                                 storing the performance of each tested model.
     run.py                    # `main()` function that selects the hyper-parameters of the model with the best performance, 
                                 trains the model on all training data and produces predictions on the test dataset.

```

## Running vanilla models
To run and assess the vanilla models, please run the following command:
```bash
python src/run_vanilla_models.py
```

## Running model selection
To run the model selection process, please run the following command:
```bash
python src/run_model_selection.py
```

## Running final model
To train the final model and test it in the testing data, please run the following command:
```bash
python src/run.py
```




