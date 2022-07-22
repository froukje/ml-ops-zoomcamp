import os

import batch

YEAR = os.getenv("YEAR", 2021)
MONTH = os.getenv("MONTH", 2)

model_service = batch.main(
    year=YEAR,
    month=MONTH
)


def lambda_handler(event, context):
    # pylint: disable=unused-argument
    return model_service.lamda_handler(event)
