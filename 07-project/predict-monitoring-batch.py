import json
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
from prefect import flow, task

from evidently import ColumnMapping

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection


@task
def upload_target(filename):

    client = MongoClient("mongodb://localhost:27018/")
    collection = client.get_database("prediction_service").get_collection("data")
    with open(filename) as f_target:
        for line in f_target.readlines():
            row = line.split(",")
            collection.update_one({"id": row[0]}, {"$set": {"target": float(row[1])}})
            data = list(collection.find())

@task
def load_reference_data(filename):

    RUN_ID = 'dd45baed06ef478e9646e1010a0b80f8'
    logged_model = f'./mlruns/1/{RUN_ID}/artifacts/models'

    model = mlflow.pyfunc.load_model(logged_model)

    dv_path = f'./mlruns/1/{RUN_ID}/artifacts/dv.bin'
    scaler_path = f'./mlruns/1/{RUN_ID}/artifacts/scaler.bin'

    with open(dv_path, 'rb') as f_out:
        dv = pickle.load(f_out)
    with open(scaler_path, 'rb') as f_out:
        scaler = pickle.load(f_out)

    reference_data = pd.read_csv(filename)

    # create features
    # define numerical and categorical features
    numerical = ["X1", "X2", "X3", "X4", "X5", "X7"]
    categorical = ["X6", "X8"]
    
    # preprocess numerical features
    X_num = scaler.transform(reference_data[numerical])
    # preprocess categorical features
    reference_data[categorical] = reference_data[categorical].astype("string")
    X_dicts = reference_data[categorical].to_dict(orient="records")
    X_cat = dv.transform(X_dicts)
    # concatenate both
    X = np.concatenate((X_num, X_cat), axis=1)

    pred = model.predict(X)[0]
    reference_data['prediction'] = pred 

    return reference_data

@task
def fetch_data():
    client = MongoClient("mongodb://localhost:27018/")
    data = client.get_database("prediction_service").get_collection("data").find()
    df = pd.DataFrame(list(data))
    return df

@task
def run_evidently(ref_data, data):
    profile = Profile(sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()])
    mapping = ColumnMapping(target="Y1", prediction="prediction", numerical_features=['X1', 'X2', 'X3', 'X4', 'X5', 'X7'],
                            categorical_features=['X6', 'X8'],
                            datetime_features=[])
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab(verbose_level=0)])
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard


@task
def save_report(result):
    client = MongoClient("mongodb://localhost:27018/")
    client.get_database("prediction_service").get_collection("report").insert_one(result[0])


@task
def save_html_report(result):
    result[1].save("evidently_report_example.html")


@flow
def batch_analyze():
    upload_target("target.csv")
    ref_data = load_reference_data("./evidently_service/datasets/ENB2012_data.csv")
    data = fetch_data()
    result = run_evidently(ref_data, data)
    save_report(result)
    save_html_report(result)

batch_analyze()
