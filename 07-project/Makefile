LOCAL_TAG:=$(shell date +"%Y-%m")
LOCAL_IMAGE_NAME:=heat-load:${LOCAL_TAG}

   test:
	pytest tests/unit_tests.py

   quality_checks:
	isort .
	black .
	pylint --recursive=y .

   build: quality_checks test 
	docker build -t ${LOCAL_IMAGE_NAME} .    
	
   integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NANE} ./integration-test/run.sh

   publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGENAME} ./scripts/publish.sh

   setup:
	pipenv install --dev
	pre-commit install



