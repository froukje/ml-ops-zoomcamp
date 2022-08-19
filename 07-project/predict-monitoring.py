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

RUN_ID = '2d5ce3ffec2d45d98fd1796654e7ba42'
logged_model = f'./mlruns/1/{RUN_ID}/artifacts/models'

#MODEL_FILE = os.getenv('MODEL_FILE', os.path.join(logged_model, 'model.xgb'))
MONGODB_ADDRESS = os.getenv('MONGODB_ADDRESS', 'mongodb://127.0.0.1.27017')
EVIDENTLY_SERVICE_ADRESSS = os.getenv('EVIDENTLY_SERVICE_', 'http://127.0.0.1:5000')

model = mlflow.pyfunc.load_model(logged_model)

dv_path = f'./mlruns/1/{RUN_ID}/artifacts/dv.bin'
scaler_path = f'./mlruns/1/{RUN_ID}/artifacts/scaler.bin'

with open(dv_path, 'rb') as f_out:
    dv = pickle.load(f_out)
with open(scaler_path, 'rb') as f_out:
    scaler = pickle.load(f_out)

app = Flask('heat-loading')
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database('prediction_serice')
collection = db.get_collection('data')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    record = request.get_json()
    
    # turn json input to dataframe
    record = pd.DataFrame([data])
    record = data.rename(columns={"X1": "relative_compactnes",
                                "X2": "surface_area",
                                "X3": "wall_area",
                                "X4": "roof_area",
                                "X5": "overall_height",
                                "X6": "orientation",
                                "X7": "glazing_area",
                                "X8": "glazing_area_distribution",
                                "Y1": "heating_load",
                                "Y2": "cooling_load"})

    # define numerical and categorical features
    numerical = ["relative_compactnes", "surface_area", "wall_area",
                 "roof_area", "overall_height", "glazing_area"]
    categorical = ["orientation", "glazing_area_distribution"]
    
    # preprocess numerical features
    X_num = scaler.transform(record[numerical])
    # preprocess categorical features
    record[categorical] = record[categorical].astype("string")
    X_dicts = record[categorical].to_dict(orient="records")
    X_cat = dv.transform(X_dicts)
    # concatenate both
    X = np.concatenate((X_num, X_cat), axis=1)

    pred = model.predict(X)
    pred = predict(features)[0]

    result = {
            'heat load': float(pred),
            'model_version': RUN_ID
            }
   

    save_to_db(record, prediction)

    send_to_evidently_service(record, prediction)

    return jsonify(result)

def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction

    collection.insert_one(rec)

def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction

    requests.post(f'{EVIDENTLY_SERVICE_ADDRESS}/iterate/heat_load', json=[recrecord, prediction

if __name__ == '__main__()':
    app.run(debug=True, host='0.0.0.0', port=9696)
