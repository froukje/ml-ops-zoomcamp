'''integration test for example input'''
import json

import requests
from deepdiff import DeepDiff

input_example = {
    "X1": 0.98,
    "X2": 514.50,
    "X3": 294.00,
    "X4": 110.25,
    "X5": 7.00,
    "X6": 2,
    "X7": 0.00,
    "X8": 0,
}


url = 'http://localhost:9696/predict'
actual_response = requests.post(url, json=input_example, timeout=5).json()

print('actual response')
print(json.dumps(actual_response, indent=2))

expected_response = {
    'heat_load': 16.4,
    'model_version': 'b03839b56eb74863ba7df86677772c25',
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff
