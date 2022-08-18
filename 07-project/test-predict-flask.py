import requests

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

#input_example = {
#        "X1": 1,
#        "X2": 514,
#        "X3": 294,
#        "X4": 110,
#        "X5": 7,
#        "X6": 2,
#        "X7": 0,
#        "X8": 0
#        }


url = 'http://localhost:9696/predict'
response = requests.post(url, json=input_example)
print(response.json())

