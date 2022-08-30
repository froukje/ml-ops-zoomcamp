'''app for making prediction of heat load (using docker)'''
import pickle

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

RUN_ID = 'b03839b56eb74863ba7df86677772c25'
logged_model = f'./mlruns/1/{RUN_ID}/artifacts/models'
model = mlflow.pyfunc.load_model(logged_model)

dv_path = f'./mlruns/1/{RUN_ID}/artifacts/dv.bin'
scaler_path = f'./mlruns/1/{RUN_ID}/artifacts/scaler.bin'

print(model)

with open(dv_path, 'rb') as f_out:
    dv = pickle.load(f_out)
with open(scaler_path, 'rb') as f_out:
    scaler = pickle.load(f_out)


def preprocess(data):
    '''preprocess the input data'''
    # turn json input to dataframe

    data = pd.DataFrame([data])

    # define numerical and categorical features
    numerical = ["X1", "X2", "X3", "X4", "X6", "X8"]
    categorical = ["X5", "X7"]

    # preprocess numerical features
    X_num = scaler.transform(data[numerical])
    # preprocess categorical features
    data[categorical] = data[categorical].astype("string")
    X_dicts = data[categorical].to_dict(orient="records")
    X_cat = dv.transform(X_dicts)
    # concatenate both
    X = np.concatenate((X_num, X_cat), axis=1)

    return X


def predict(X):
    '''make predictions'''
    pred = model.predict(X)
    print('prediction', pred[0])
    return float(pred[0])


app = Flask('heat-loading')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    '''get data and make predictions'''
    input_data = request.get_json()
    print("INPUT", input_data)
    features = preprocess(input_data)
    pred = predict(features)

    result = {'heat load': pred, 'model_version': RUN_ID}

    return jsonify(result)


if __name__ == '__main__()':
    app.run(debug=True, host='0.0.0.0', port=9696)
