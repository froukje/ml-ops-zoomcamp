"""
Script to train Energy Efficiency model to predict heating load
Author: Frauke Albrecht
"""

import os
import pickle
from collections import namedtuple
from datetime import timedelta

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from optuna.samplers import TPESampler
from prefect import flow, task
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.task_runners import SequentialTaskRunner
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@task
def read_data(args):
    # pylint: disable=duplicate-code
    '''read input data and rename column names'''
    data = pd.read_csv(args.input_data)
    return data


@task
def onehot(data_train, data_val, categorical):
    '''one hot encoding of categorical features'''
    # pylint: disable=duplicate-code

    # change data type from integer to string for categorical features
    data_train[categorical] = data_train[categorical].astype("string")
    data_val[categorical] = data_val[categorical].astype("string")
    train_dicts = data_train[categorical].to_dict(orient="records")
    val_dicts = data_val[categorical].to_dict(orient="records")
    dv = DictVectorizer(sparse=False)  # don't use sparse matrix
    dv.fit(train_dicts)
    X_train_cat = dv.transform(train_dicts)
    X_val_cat = dv.transform(val_dicts)

    return X_train_cat, X_val_cat, dv


@task
def normalize(data_train, data_val, numerical):
    '''normalize numerical features'''
    # pylint: disable=duplicate-code
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(data_train[numerical])
    X_val_num = scaler.transform(data_val[numerical])
    return X_train_num, X_val_num, scaler


@task
def training(X_train, X_val, y_train, y_val, dv, scaler, args):
    '''training the model with hyperparameter tuning'''
    # pylint: disable=duplicate-code
    def objective(trial):
        # pylint: disable=duplicate-code
        mlflow.set_experiment("xgb-hyper")
        with mlflow.start_run():
            n_estimators = trial.suggest_int(
                'n_estimators', args.n_estimators[0], args.n_estimators[1], log=True
            )
            eta = trial.suggest_float('eta', args.eta[0], args.eta[1], log=True)
            gamma = trial.suggest_float('gamma', args.gamma[0], args.gamma[1])
            alpha = trial.suggest_float('alpha', args.alpha[0], args.alpha[1])
            max_depth = trial.suggest_categorical('max_depth', args.max_depth)
            min_child_weight = trial.suggest_categorical(
                'min_child_weight', args.min_child_weight
            )
            params = {
                "objective": 'reg:squarederror',
                "n_estimators": n_estimators,
                "eta": eta,
                "gamma": gamma,
                "alpha": alpha,
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "verbosity": 1,
            }

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train).reshape(-1, 1)
            rmse_train = mean_squared_error(y_pred_train, y_train, squared=False)
            mlflow.log_metric('rmse_train', rmse_train)
            y_pred_val = model.predict(X_val).reshape(-1, 1)
            rmse_val = mean_squared_error(y_pred_val, y_val, squared=False)
            mlflow.log_metric('rmse_val', rmse_val)
            mlflow.log_params(params)
            # save scaler and dv as artifacts
            path = os.path.join(args.output, 'dv.bin')
            with open(path, 'wb') as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact(local_path=path)
            path = os.path.join(args.output, 'scaler.bin')
            with open(path, 'wb') as f_out:
                pickle.dump(scaler, f_out)
            mlflow.log_artifact(local_path=path)

            # save model
            mlflow.xgboost.log_model(model, artifact_path="models")
            # save run id
            run_id = mlflow.active_run().info.run_id
            trial.set_user_attr('run_id', run_id)
            return rmse_val

    study = optuna.create_study(sampler=TPESampler(), direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    return study


@flow(task_runner=SequentialTaskRunner())
def main(input_data='data/ENB2012_data.csv', output='output'):
    """main function to train the model"""
    # pylint: disable=no-member
    # pylint: disable=duplicate-code
    args_dict = {}
    args_dict["input_data"] = input_data
    args_dict["output"] = output
    args_dict["n_estimduplicate-codeators"] = [500, 1000]
    args_dict["max_depth"] = [5, 10, 100, None]
    args_dict["min_samples_leaf"] = [1, 10, 50]
    args_dict["eta"] = [0.1, 0.5]
    args_dict["gamma"] = [0, 1]
    args_dict["alpha"] = [0, 1]
    args_dict["min_child_weight"] = [1, 10, 50]
    args_dict["n_trials"] = 200

    args = namedtuple("ObjectName", args_dict.keys())(*args_dict.values())
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    client = MlflowClient("http://127.0.0.1:5000")

    # read data
    data = read_data(args).result()

    # define features and target
    features = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    target = "Y1"
    # define numerical and categorical features
    numerical = ["X1", "X2", "X3", "X4", "X5", "X7"]
    categorical = ["X6", "X8"]

    # train-/ val-/ test split
    data_train_full, data_test = train_test_split(
        data[features + [target]], test_size=0.2, random_state=42
    )
    data_train, data_val = train_test_split(
        data_train_full[features + [target]], test_size=0.25, random_state=42
    )

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
    X_train_num, X_val_num, scaler = normalize(data_train, data_val, numerical).result()
    ## one-hot encode categorical variables
    X_train_cat, X_val_cat, dv = onehot(data_train, data_val, categorical).result()

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
    mlflow.register_model(model_uri=f"runs:/{run_id}/models", name='heat-load')
    print("Registered Models:")
    print(client.list_registered_models())


DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner=SubprocessFlowRunner(),
    tags=["ds-train"],
)
