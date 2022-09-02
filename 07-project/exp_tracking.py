'''Script to train Energy Efficiency model to predict heating load
Author: Frauke Albrecht'''

import argparse
import os
import pickle

import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from optuna.samplers import TPESampler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_data(args_params):
    '''read input data and rename column names'''
    data = pd.read_csv(args_params.input_data)
    return data


def onehot(data_train, data_val, categorical):
    '''one hot encoding of categorical features'''

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


def normalize(data_train, data_val, numerical):
    '''normalize numerical features'''
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(data_train[numerical])
    X_val_num = scaler.transform(data_val[numerical])

    return X_train_num, X_val_num, scaler


def training(X_train, X_val, y_train, y_val, dv, scaler):
    '''training the model with hyperparameter tuning'''

    def objective(trial):
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
            y_pred_val = model.predict(X_val).reshape(-1, 1)
            rmse_train = mean_squared_error(y_pred_train, y_train, squared=False)
            rmse_val = mean_squared_error(y_pred_val, y_val, squared=False)

            mlflow.log_params(params)
            mlflow.log_metric('rmse_train', rmse_train)
            mlflow.log_metric('rmse_val', rmse_val)

            # save scaler and dv as artifacts
            dv_name = 'dv.bin'
            scaler_name = 'scaler.bin'
            dv_path = os.path.join(args.output, dv_name)
            scaler_path = os.path.join(args.output, scaler_name)
            with open(dv_path, 'wb') as f_out:
                pickle.dump(dv, f_out)
            with open(scaler_path, 'wb') as f_out:
                pickle.dump(scaler, f_out)
            mlflow.log_artifact(local_path=dv_path)
            mlflow.log_artifact(local_path=scaler_path)

            # save model
            mlflow.xgboost.log_model(model, artifact_path="models")

            # save run id
            run = mlflow.active_run()
            run_id = run.info.run_id
            trial.set_user_attr('run_id', run_id)

            return rmse_val

    study = optuna.create_study(sampler=TPESampler(), direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)

    return study


def main(args_params):
    '''
    main function for to train the model
    '''

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    client = MlflowClient("http://127.0.0.1:5000")

    # read data
    data = read_data(args_params)

    # define features and target
    features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    target = 'Y1'
    # define numerical and categorical features
    numerical = ['X1', 'X2', 'X3', 'X4', 'X5', 'X7']
    categorical = ['X6', 'X8']

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
    X_train_num, X_val_num, scaler = normalize(data_train, data_val, numerical)
    ## one-hot encode categorical variables
    X_train_cat, X_val_cat, dv = onehot(data_train, data_val, categorical)

    # concatenate numerical and categorical features
    X_train = np.concatenate((X_train_num, X_train_cat), axis=1)
    X_val = np.concatenate((X_val_num, X_val_cat), axis=1)
    print(f"train data shape after preprocessing {X_train.shape}")
    print(f"val data shape after preprocessing {X_train.shape}")

    # define the target
    y_train = data_train[target]
    y_val = data_val[target]

    # train the model with hyperparameter tuning
    study = training(X_train, X_val, y_train, y_val, dv, scaler)
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

    mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data locations
    parser.add_argument('--input-data', type=str, default='data/ENB2012_data.csv')
    parser.add_argument('--output', type=str, default='output')
    # model parameters
    parser.add_argument('--n-estimators', type=int, nargs='+', default=[500, 1000])
    parser.add_argument('--max-depth', type=int, nargs='+', default=[5, 10, 100, None])
    parser.add_argument('--max-depth-none', action='store_true', default=False)
    parser.add_argument('--min-samples-leaf', type=int, nargs='+', default=[1, 10, 50])
    parser.add_argument(
        '--eta', type=float, nargs='+', default=[0.1, 0.5]
    )  # default 0.3
    parser.add_argument('--gamma', type=float, nargs='+', default=[0, 1])  # default=0
    parser.add_argument('--alpha', type=float, nargs='+', default=[0, 1])  # default=0
    parser.add_argument('--min-child_weight', type=int, nargs='+', default=[1, 10, 50])
    # nr of trials for hyperparameter tuning
    parser.add_argument('--n-trials', type=int, default='20')
    args = parser.parse_args()

    # None is added to max-depth (cannot be done directly -> type error)
    if args.max_depth_none:
        args.max_depth = args.max_depth + [None]

    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print()

    main(args)
