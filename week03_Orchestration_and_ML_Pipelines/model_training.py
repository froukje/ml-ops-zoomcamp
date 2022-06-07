from asyncio import tasks
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error
import mlflow
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def add_features(train_path, val_path):
    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    print(f'train data length: {len(df_train)}')
    print(f'val data: {len(df_val)}')

    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv

@task
def train_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            # log paramters in mlflow
            mlflow.log_params(params)
            booster = xgb.train(
                # paramters are passed to xgboost
                params=params,
                # training on train data
                dtrain=train,
                # set boosting rounds
                num_boost_round=100,
                # validation is done on validation dataset
                evals=[(valid, 'validation')],
                # if model does not improve for 50 methods->stop
                early_stopping_rounds=50
            )
            # make predictions
            y_pred = booster.predict(valid)
            # calculate error
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            # log metric
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    # define the search space, i.e. the range of parameters for hyperparamter tuning
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0), #[exp(-3), exp(0)] = [0.05, 1]
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42  
    }

    # fmin method tries to minimize the metric
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials() 
    )

def train_best_model(train, valid, y_val, dv):
    with mlflow.start_run():
        best_params =  {'learning_rate': 0.20905792515510074,
                        'max_depth': 7,
                        'min_child_weight': 0.5241500975917085,
                        'objective': 'reg:squarederror',
                        'reg_alpha': 0.13309121698466933,
                        'reg_lambda': 0.11277257081373988,
                        'seed': 42}
        mlflow.log_params(best_params)

        booster = xgb.train(
                    # paramters are passed to xgboost
                    params=best_params,
                    # training on train data
                    dtrain=train,
                    # set boosting rounds
                    num_boost_round=100,
                    # validation is done on validation dataset
                       evals=[(valid, 'validation')],
                    # if model does not improve for 50 methods->stop
                    early_stopping_rounds=50
                )


        # make predictions
        y_pred = booster.predict(valid)
        # calculate error
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        # log metric
        mlflow.log_metric("rmse", rmse)

        # save preprocessor
        with open("../week02/models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        # log preprocessor
        mlflow.log_artifacts("../week02/models", artifact_path="preprocessor")
        # log the model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

@flow(task_runner=SequentialTaskRunner())
def main(train_path='../data/green_/home/frauketripdata_2021-01.parquet', 
         val_path='../data/green_tripdata_2021-02.parquet'):

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("nyc-taxi-experiment-2")
    X_train, X_val, y_train, y_val, dv = add_features(train_path, val_path).result()
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    train_model_search(train, valid, y_val)
    train_best_model(train, valid, y_val, dv)

if __name__ == "__main__":
    main()

