#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import numpy as np


def read_data(filename):
    df = pd.read_parquet(filename)
    
    print(df.head())
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def apply_model(df):

    with open('model.bin', 'rb') as f_in:
       dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    y_pred = lr.predict(X_val)
    print(f'mean duration: {np.mean(y_pred)}')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicstions'] = y_pred

    return df_result

def run():
    year = 2021# int(sys.argv[1])#2021
    month = 4#int(sys.argv[2])#2
    print(f"read data for year {year} and month {month}...") 
    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    output_file = f"output/homework4-{year:04d}-{month:02d}.parquet"#"../../output/homework_04_q2.parquet"

    print("apply the model ...")
    df_result = apply_model(df) 

    print("save the results...")
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    categorical = ['PUlocationID', 'DOlocationID']
    run()


