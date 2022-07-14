#!/usr/bin/env bash

cd "$(dirname "$0")" # cd to directory of the script

LOCAL_TAG='date + "%Y-%m-%d-%H-%M"' # gives the current date
export LOCAL_IMAGE_NAME="stram-model-duration:${LOCAL_TAG}"
export PREDICTION_STREAM_NAME="ride_predictions"

docker build -t ${LOCAL_IMAGE_NAME} ..

docker-compose up -d

sleep 1

aws endpoind-url=http://localhost:4566 \
	kinesis create stream \
	--stream-name ride-predictions \
	--shard-count-value 1

pipenv run python test_docker.py

ERROR_CODE=$?
if [ ${ERROR_CODE} != 0]; then
    docker-compose logs
fi

docker-compose down
exit ${ERROR_CODE}
