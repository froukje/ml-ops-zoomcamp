# Script to train Energy Efficiency model to predict heating load
# Author: Frauke Albrecht

import os
import pandas as pd
import numpy as np
import argparse
import pickle
from datetime import datetime
from collections import namedtuple

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

import mlflow
from mlflow.tracking import MlflowClient

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

@task
def read_data(args):
    ''' read input data and rename column names'''
    data = pd.read_csv(args.input_data)
    data = data.rename(columns={"X1": "relative_compactnes", 
                                "X2": "surface_area",
                                "X3": "wall_area", 
                                "X4": "roof_area", 
                                "X5": "overall_height",
                                "X6": "orientation", 
                                "X7": "glazing_area", 
                                "X8": "glazing_area_distribution",
                                "Y1": "heating_load", 
                                "Y2": "cooling_load"})
    return data

@task
def onehot(args, data_train, data_val, categorical):
    '''one hot encoding of categorical features'''

    # change data type from integer to string for categorical features
    data_train[categorical] = data_train[categorical].astype("string")
    data_val[categorical] = data_val[categorical].astype("string")
    train_dicts = data_train[categorical].to_dict(orient="records")
    val_dicts = data_val[categorical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False) # don't use sparse matrix
    dv.fit(train_dicts)
    X_train_cat = dv.transform(train_dicts)
    X_val_cat = dv.transform(val_dicts)

    return X_train_cat, X_val_cat, dv

@task
def normalize(args, data_train, data_val, numerical):
    ''' normalize numerical features'''
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(data_train[numerical])
    X_val_num = scaler.transform(data_val[numerical])
   
    return X_train_num, X_val_num, scaler

@task
def training(X_train, X_val, y_train, y_val, dv, scaler, args):
    '''training the model with hyperparameter tuning'''
    def objective(trial):
        mlflow.set_experiment("xgb-hyper")
        with mlflow.start_run():
            n_estimators =  trial.suggest_int('n_estimators', args.n_estimators[0], args.n_estimators[1], log=True)
            eta = trial.suggest_float('eta', args.eta[0], args.eta[1], log=True)
            gamma = trial.suggest_float('gamma', args.gamma[0], args.gamma[1])
            alpha = trial.suggest_float('alpha', args.alpha[0], args.alpha[1])
            max_depth = trial.suggest_categorical('max_depth', args.max_depth)
            min_child_weight = trial.suggest_categorical('min_child_weight', args.min_child_weight)
        
            params = {"objective": 'reg:squarederror',
                      "n_estimators": n_estimators,
                      "eta": eta,
                      "gamma": gamma,
                      "alpha": alpha,
                      "max_depth": max_depth,
                      "min_child_weight": min_child_weight,
                      "verbosity": 1}

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train).reshape(-1,1)
            y_pred_val = model.predict(X_val).reshape(-1,1)
            rmse_train = mean_squared_error(y_pred_train, y_train, squared=False)
            rmse_val = mean_squared_error(y_pred_val, y_val, squared=False)

            mlflow.log_params(params)
            mlflow.log_metric('rmse_train', rmse_train)
            mlflow.log_metric('rmse_val', rmse_val)

            # save scaler and dv as artifacts
            dv_name = f'dv.bin'
            scaler_name = f'scaler.bin'
            dv_path = os.path.join(args.output, dv_name)
            scaler_path = os.path.join(args.output, scaler_name)
            with open(dv_path, 'wb') as f_out:
                pickle.dump(dv, f_out)
            with open(scaler_path, 'wb') as f_out:
                pickle.dump(scaler, f_out)
            mlflow.log_artifact(local_path=dv_path)
            mlflow.log_artifact(local_path=scaler_path)

            # save model
            mlflow.xgboost.log_model(model, artifact_path="models_mlflow")
            
            # save run id
            run = mlflow.active_run()
            run_id = run.info.run_id
            trial.set_user_attr('run_id', run_id)
        
            return rmse_val

    study = optuna.create_study(sampler=TPESampler(), direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    
    return study

@flow(task_runner=SequentialTaskRunner())
def main(input_data='data/ENB2012_data.csv',
         output='output'):

    args_dict = {}
    args_dict["input_data"] = input_data
    args_dict["output"] = output
    args_dict["n_estimators"] = [500, 1000]
    args_dict["max_depth"] = [5, 10, 100, None]
    args_dict["min_samples_leaf"] = [1, 10, 50]
    args_dict["eta"] = [0.1, 0.5]
    args_dict["gamma"] = [0, 1]
    args_dict["alpha"] = [0, 1]
    args_dict["min_child_weight"] = [1, 10, 50]
    args_dict["n_trials"] = 2

    args = namedtuple("ObjectName", args_dict.keys())(*args_dict.values())
    

    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    client = MlflowClient("http://127.0.0.1:5000")

    # read data
    data = read_data(args).result()

    # define features and target
    features = ["relative_compactnes", "surface_area", "wall_area", 
                "roof_area", "overall_height", "orientation", "glazing_area", 
                "glazing_area_distribution"] 
    target = "heating_load"
    # define numerical and categorical features
    numerical = ["relative_compactnes", "surface_area", "wall_area", 
                 "roof_area", "overall_height", "glazing_area"]
    categorical = ["orientation", "glazing_area_distribution"] 

    # train-/ val-/ test split
    data_train_full, data_test = train_test_split(data[features+[target]], test_size=0.2, random_state=42)
    data_train, data_val = train_test_split(data_train_full[features+[target]], test_size=0.25, random_state=42)

    data_train_full = data_train_full.reset_index(drop=True)
    data_train = data_train.reset_index(drop=True)
    data_val = data_val.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    # check dataframes
    print(f"train data length {len(data_train)}")
    print(f"val data length {len(data_val)}")
    print(f"test data length {len(data_test)}")
   
    # preprocessing
    ## normalize numerical variables
    X_train_num, X_val_num, scaler = normalize(args, data_train, data_val, numerical).result()
    ## one-hot encode categorical variables 
    X_train_cat, X_val_cat, dv = onehot(args, data_train, data_val, categorical).result()

    # concatenate numerical and categorical features
    X_train = np.concatenate((X_train_num, X_train_cat), axis=1)
    X_val = np.concatenate((X_val_num, X_val_cat), axis=1)
    print(f"train data shape after preprocessing {X_train.shape}")
    print(f"val data shape after preprocessing {X_train.shape}")

    # define the target
    y_train = data_train[target]
    y_val = data_val[target]
    
    # train the model with hyperparameter tuning
    study = training(X_train, X_val, y_train, y_val, dv, scaler, args).result()
    print("Number of finished trials: ", len(study.trials))

    # save best parameters
    best_params = study.best_params
    print("Best model parameters:")
    print(best_params)
    attr = study.best_trial.user_attrs
    print("Best model mlflow run id:")
    print(attr)

    # register best model
    run_id = attr["run_id"]
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/models",
        name='heat-load'
    )
    print("Registered Models:")
    print(client.list_registered_models())



    #main(args)
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
        flow=main,
        name="model_training",
        schedule=IntervalSchedule(interval=timedelta(minutes=5)),
        flow_runner=SubprocessFlowRunner(),
        tags=["ds-train"]
        )
