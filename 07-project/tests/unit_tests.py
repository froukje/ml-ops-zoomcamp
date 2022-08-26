import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import trainingtests

def test_to_string():
    '''test for function 'onehot' in trainingtests.py'''
    # example input
    input_example = {
        "X1": 0.98,
        "X2": 514.50,
        "X3": 294.00,
        "X4": 110.25,
        "X5": 7.00,
        "X6": 2,
        "X7": 0.00,
        "X8": 0
        }
    input_example = pd.DataFrame([input_example])
    categorical = ["X6", "X8"]

    actual = trainingtests.cat_to_string(input_example, categorical)[categorical]

    expected = input_example[categorical].astype("string")
    
    assert (expected.dtypes == actual.dtypes).all, "error in type transformation of cat variables"

def test_concatenate():
    '''test for function 'concatenate' in trainingtests.py'''
    # example input
    input_example = {
        "X1": 0.98,
        "X2": 514.50,
        "X3": 294.00,
        "X4": 110.25,
        "X5": 7.00,
        "X6": 2,
        "X7": 0.00,
        "X8": 0
        }
    input_example = pd.DataFrame([input_example])
    numerical = ["X1", "X2", "X3", "X4", "X5", "X7"]
    categorical = ["X6", "X8"]
    X_num = input_example[numerical]
    X_cat = input_example[categorical]

    actual = trainingtests.concatenate(X_num, X_cat)
    expected = np.concatenate((X_num, X_cat), axis=1)
    
    assert len(actual) == len(expected), "error in concatenating numerical and categorical features"
