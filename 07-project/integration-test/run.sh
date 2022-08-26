#!/usr/bin/env bash

# go to directory `integration-test`
cd "(dirname "$0")"

LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
export LOCAL_IMAGE_NAME="heat-load:${LOCAL_TAG}"
docker build -t ${LOCAL_IMAGE_NAME} ..

cd integration-test
docker-compose up --build -d

sleep 1

pipenv run python integration_test_predict.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
fi

docker-compose down

exit ${ERROR_CODE}
