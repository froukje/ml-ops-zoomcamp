import os
import model


PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')
RUN_ID = os.getenv('RUN_ID')
#logged_model = f's3://mlflow-models-alexey/1/{RUN_ID}/artifacts/model'
#logged_model = f'runs:/{RUN_ID}/model'
#model = mlflow.pyfunc.load_model(logged_model)
TEST_RUN = os.getenv('TEST_RUN', 'False') == 'True'

model_service = model.init(
    prediction_stream_name=PREDICTIONS_STREAM_NAME,
    run_id=RUN_ID,
    test_run=TEST_RUN
)

def lambda_handler(event, context):
    return model_service.lamda_handler(event)