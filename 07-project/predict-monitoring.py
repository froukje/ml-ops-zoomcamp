import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from flask import Flask, request, jsonify
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np
from pymongo import MongoClient
import requests

RUN_ID = 'dd45baed06ef478e9646e1010a0b80f8'
logged_model = f'./mlruns/1/{RUN_ID}/artifacts/models'

MONGODB_ADDRESS = os.getenv('MONGODB_ADDRESS', 'mongodb://127.0.0.1.27017')
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')

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

    result = {
            'heat_load': float(pred),
            'model_version': RUN_ID
            }
   

    record = record.to_dict(orient="index")[0]
    save_to_db(record, float(pred))

    #send_to_evidently_service(record, float(pred))

    return jsonify(result)

def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction

    print('RECORD', rec)
    collection.insert_one(rec)

def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction

    requests.post(f'{EVIDENTLY_SERVICE_ADDRESS}/iterate/heat_load', json=[rec])

if __name__ == '__main__()':
    app.run(debug=True, host='0.0.0.0', port=9696)
