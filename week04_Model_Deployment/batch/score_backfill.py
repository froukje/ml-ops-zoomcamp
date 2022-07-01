import score_prefect
from prefect import flow
from dateutil.relativedelta import relativedelta
from datetime import datetime

@flow
def ride_duration_prediction_backfill():
    start_date = datetime(year=2021, month=3, day=1) 
    end_date = datetime(year=2022, month=4, day=1) 

    d = start_date

    while d <= end_date:
        score_prefect.ride_duration_prediction(run_id="54c0663c677e417f8900b51ed7985878", run_date=d)

        d = d + relativedelta(months=1)

if __name__ == '__main__':
    ride_duration_prediction_backfill()
