import pickle
import pandas as pd
import datetime

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    logger = get_run_logger()    
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")

    return 

@task
def get_paths(date_=None):
    if date_==None:
        date_ = datetime.date.today()
    else:
        date_ = datetime.datetime.strptime(date_, '%Y-%m-%d')

    # get data of previous month
    first = date_.replace(day=1)
    lastMonth = first - datetime.timedelta(days=1)
    lastMonth_str = lastMonth.strftime("%Y-%m")
    val_path = "../../data/fhv_tripdata_"+lastMonth_str+".parquet"
    # get data of two previous months
    first = lastMonth.replace(day=1)
    twoMonth = first - datetime.timedelta(days=1)
    twoMonth = twoMonth.strftime("%Y-%m")
    train_path = "../../data/fhv_tripdata_"+twoMonth+".parquet"
    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(date_="2021-08-15"):

    train_path, val_path = get_paths(date_).result()
    print(train_path, val_path)
    
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    
    # save preprocessor
    with open(f"../../week02_Experiment_tracking_and_model_management/models/dv-{date_}.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)
    # save model
    with open(f"../../week02_Experiment_tracking_and_model_management/models/model-{date_}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)


main()
