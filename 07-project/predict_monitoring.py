'''make predictions and save them to db'''
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from pymongo import MongoClient

RUN_ID = 'b03839b56eb74863ba7df86677772c25'
logged_model = f'./mlruns/1/{RUN_ID}/artifacts/models'

MONGODB_ADDRESS = os.getenv('MONGODB_ADDRESS', 'mongodb://127.0.0.1.27017')

model = mlflow.pyfunc.load_model(logged_model)

dv_path = f'./mlruns/1/{RUN_ID}/artifacts/dv.bin'
scaler_path = f'./mlruns/1/{RUN_ID}/artifacts/scaler.bin'

with open(dv_path, 'rb') as f_out:
    dv = pickle.load(f_out)
with open(scaler_path, 'rb') as f_out:
    scaler = pickle.load(f_out)

app = Flask('heat_load')
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database('prediction_service')
collection = db.get_collection('data')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    '''request data and make predictions'''
    record = request.get_json()

    # turn json input to dataframe
    record = pd.DataFrame([record])

    # define numerical and categorical features
    numerical = ["X1", "X2", "X3", "X4", "X5", "X7"]
    categorical = ["X6", "X8"]

    # preprocess numerical features
    X_num = scaler.transform(record[numerical])
    # preprocess categorical features
    record[categorical] = record[categorical].astype("string")
    X_dicts = record[categorical].to_dict(orient="records")
    X_cat = dv.transform(X_dicts)
    # concatenate both
    X = np.concatenate((X_num, X_cat), axis=1)

    pred = model.predict(X)[0]

    result = {'heat_load': float(pred), 'model_version': RUN_ID}

    record = record.to_dict(orient="index")[0]
    save_to_db(record, float(pred))

    return jsonify(result)


def save_to_db(record, prediction):
    '''save predictions to db'''
    rec = record.copy()
    rec['prediction'] = prediction

    print('RECORD', rec)
    collection.insert_one(rec)


if __name__ == '__main__()':
    app.run(debug=True, host='0.0.0.0', port=9696)
