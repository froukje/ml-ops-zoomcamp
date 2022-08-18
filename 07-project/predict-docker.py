import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from flask import Flask, request, jsonify
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np

#MLFLOW_TRACKING_URI = "http://127.0.0.1:5005"
#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
#print("Registered Models:")
#print(client.list_registered_models())

# specify which model to load
#model_name = "heat-load"
#model_version = 2

# load the model from the registry
#model = mlflow.pyfunc.load_model(
#    model_uri=f"models:/{model_name}/{model_version}"
#)
#print(f'model loaded: {model}')

# get run_id
#for mv in client.search_model_versions("name='heat-load'"):
#    mv = dict(mv)
#    if mv['version'] == str(model_version):
#        RUN_ID = mv['run_id']


# dv, scaler, parameters
#dv_path = "dv.bin"
#scaler_path = "scaler.bin"
#dv_path = client.download_artifacts(run_id=RUN_ID, path=dv_path)
#scaler_path = client.download_artifacts(run_id=RUN_ID, path=scaler_path)
#print(f'downloaded dict vectorizer and scaler to {dv_path} and {scaler_path}')

RUN_ID = '2d5ce3ffec2d45d98fd1796654e7ba42'
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

    # turn json input to dataframe

    data = pd.DataFrame([data])
    data = data.rename(columns={"X1": "relative_compactnes",
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
    X_num = scaler.transform(data[numerical])
    # preprocess categorical features
    data[categorical] = data[categorical].astype("string")
    X_dicts = data[categorical].to_dict(orient="records")
    X_cat = dv.transform(X_dicts)
    # concatenate both
    X = np.concatenate((X_num, X_cat), axis=1)

    return X

def predict(X):

    pred = model.predict(X)
    print('prediction', pred[0])
    return float(pred[0])

app = Flask('heat-loading')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    input_data = request.get_json()
    print("INPUT", input_data)
    features = preprocess(input_data)
    pred = predict(features)

    result = {
            'heat load': pred,
            'model_version': RUN_ID
            }
   
    return jsonify(result)

if __name__ == '__main__()':
    app.run(debug=True, host='0.0.0.0', port=9696)
