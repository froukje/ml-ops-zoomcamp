#!/usr/bin/env bash

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then

	LOCAL_TAG=`date +"%Y-%m"`
	export LOCAL_IMAGE_NAME="heat-load:${LOCAL_TAG}"
	echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
	docker build -t ${LOCAL_IMAGE_NAME} ..

else
	echo "no need to build ${LOCAL_IMAGE_NAME}"
fi

cd "$(dirname "$0")"
echo "$PWD"
echo "The script you are running has basename $( basename -- "$0"; ), dirname $( dirname -- "$0"; )";
echo "The present working directory is $( pwd; )";
docker-compose up --build -d

sleep 5

pipenv run python integration_test_predict.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
fi

docker-compose down

exit ${ERROR_CODE}
