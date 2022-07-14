```docker build -t stream-model-duration:v2 .```

```
docker run -it --rm \
    -p 8080:8080 \
    -e PREDICTIONS_STREAM_NAME="ride_prediction" \
    -e RUN_ID="e1efc53e9bd149078b0c12aeaa6365df" \
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="eu-west-1" \
    stream-model-duration:v2
```

```
docker run -it --rm \
    -p 8080:8080 \
    -e PREDICTION_STREAM_NAME="ride_predcitions" \
    -e RUN_id = "e1efc53e9bd149078b0c12aeaa6365df" \
    -e MODEL_LOCATION="/app/model/" \ 
    -e TEST_RUN="True" \
    -e AWS_DEFAULT_REGION="eu-west-1" \
    -v $(pwd)/model:/app/model \
    stream-model-duration:v2 
```

```
aws endpoind-url=http://localhost:4566 kinesis list-streams
```

