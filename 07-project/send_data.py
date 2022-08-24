import json
import uuid
from datetime import datetime
from time import sleep

import pandas as pd
import requests

data = pd.read_csv("data/ENB2012_data.csv")

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


with open("target.csv", 'w') as f_target:
    for index, row in data.iterrows():
        row['id'] = str(uuid.uuid4())
        f_target.write(f"{row['id']},{row['Y1']}\n")
        row = row.to_json()
        resp = requests.post("http://127.0.0.1:9696/predict",
                             headers={"Content-Type": "application/json"},
                             data=row).json()
        print(f"prediction: {resp['heat_load']}")
        sleep(1)
