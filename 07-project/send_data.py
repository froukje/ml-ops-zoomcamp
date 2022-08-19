import json
import uuid
from datetime import datetime
from time import sleep

import pandas as pd
import requests

data = pd.read_csv("data/ENB2012_data.csv")
#table = pd.read_table("data/ENB2012_data.csv")
#data = table.to_pylist()
print(data.head())

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


with open("target.csv", 'w') as f_target:
    for i in range(len(data)):
        row = data.iloc[i]
        row['id'] = str(uuid.uuid4())
        f_target.write(f"{row['id']}\n")
        row = row.to_json()
        resp = requests.post("http://127.0.0.1:9696/predict",
                             headers={"Content-Type": "application/json"},
                             data=row).json()
        print(f"prediction: {resp['heat load']}")
        sleep(1)
