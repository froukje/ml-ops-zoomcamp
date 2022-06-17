#!/usr/bin/env python
# coding: utf-8

import os
import sys

import pickle
import uuid

import pandas as pd
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline



MLFLOW_TRACKING_URI = 'sqlite:///mlruns.db'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# create unique ids
def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids
    
def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() /60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = generate_uuids(len(df))

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df


def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def load_model(run_id):
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

def apply_model(input_file, run_id, output_file):
    print(f'reading the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    
    print(f'loading the model with RUN_ID={run_id}...')
    model = load_model(run_id)

    print(f'applying the model...')
    y_pred = model.predict(dicts)
    
    print(f'saving the results to {output_file}...')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id
    
    df_result.to_parquet(output_file, index=False)

def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3

    #RUN_ID = '54c0663c677e417f8900b51ed7985878'
    run_id = sys.argv[3] 

    #input_file = '../../data/green_tripdata_2021-01.parquet'
    input_file = f'https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_{year:04}-{month:02d}.parquet'
    output_file = f'../../output/green_tripdata_{year:04}-{month:02d}.parquet'


    apply_model(input_file=input_file, run_id=run_id, output_file=output_file)

if __name__ == '__main__':
    run()
