{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "03f84789",
   "metadata": {},
   "source": [
    "# Best Practices\n",
    "\n",
    " Build on the Lambda/kinetis example from module 4\n",
    " * add tests to the code (unit tests, integration tests), we will use the library ```pytest``` for that\n",
    " * ```pipenv install --dev pytest```, it is only needed for development\n",
    " * For working in Visual Studio:\n",
    "     * Select Python Interpreter > View > Command Pallete > \"Select: Python Interpreter\"\n",
    "     * Go in terminal to the current folder and type ```pipenv --venv``` to get the python environment of the virtual environment (```<path>/.local/share/virtualenvs/code-AxO42iuz```\n",
    "     * Copy the name ```<path>/.local/share/virtualenvs/code-AxO42iuz/bin/python``` to the Command Pallete and choose this interpreter\n",
    "     * We then get a new icon on the left hand side for testing\n",
    "     * Add the test path ```<path>/.local/share/virtualenvs/code-AxO42iuz/bin/pytest``` through \"configure tests\"\n",
    " * **Unit tests** only test small units/fractions of the code, **Integration tests**, test the entire code"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a7fcda2",
   "metadata": {},
   "source": [
    "## Unit Tests\n",
    "* Create a first test\n",
    "    * In the test folder, we need to creat a file ```__init__.py```, so that python knows that this is a python package\n",
    "    * create a file ```model_test.py``` and ```model.py```\n",
    "    * Test it using docker, we need to add the new created ```model.py``` script to te docker file: ```bash docker build -t stream-model-duration:v2 .```\n",
    "     \n",
    "```\n",
    "docker run -it --rm \\\n",
    "    -p 8080:8080 \\\n",
    "    -e PREDICTIONS_STREAM_NAME=\"ride_prediction\" \\\n",
    "    -e TEST_RUN=\"True\" \\\n",
    "    -e AWS_DEFAULT_REGION=\"eu-west-1\" \\\n",
    "    stream-model-duration:v2\n",
    "```\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d99aaeb",
   "metadata": {},
   "source": [
    "* To run the tests from the terminal: go to folder ```code```, start virtual env: ```pipenv shell```, then run ```pipenv run pytest tests/``` "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dc66bee7",
   "metadata": {},
   "source": [
    "## Integration Test with docker Compose\n",
    "\n",
    "* We use the file ```test_docker.py``` and add the test at the end:\n",
    "```\n",
    "actual_response = response = requests.post(url, json=event)\n",
    "print('actual_response')\n",
    "print(json.dumps(actual_response, indent=2)\n",
    "```\n",
    "```\n",
    "expected_response = [{\n",
    "    'predictions':  [{\n",
    "        'model': 'ride_duration_prediction_model',\n",
    "        'version': 'e1efc53e9bd149078b0c12aeaa6365df',#run_id\n",
    "        'prediction': {\n",
    "            'ride_duration': 21.294545348333408,\n",
    "            'ride_id': 256\n",
    "            }\n",
    "        }]\n",
    "    }]\n",
    "\n",
    "```\n",
    "* In order to compare the two outcoming dictionaries we use the library ```deepdiff```: ```pipenv install --dev deepdiff```\n",
    "* To compare floats only up a certain digit we can set a tolerance in deepdiff"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84f14087",
   "metadata": {},
   "source": [
    "* We now use docker-compose instead of docker\n",
    "    * We create a docker-compose.yaml\n",
    "    * We create a run.sh script\n",
    "    * With ```docker-compose up -d``` we can run the container in a detached mode, i.e. we can use the terminal after execution\n",
    "    * To stop the execution run ```docker-compose down```"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "526fb77a",
   "metadata": {},
   "source": [
    "## Testing Cloud Services with Localstack\n",
    "\n",
    "* So far we didn't test the kinesis connection (```class KinesisCallback``` in ```model.py```)\n",
    "* We will use localstack for that\n",
    "    * \"Fully functional AWS cloud stack\"\n",
    "    * We will use docker-compose to run it and integrate it in our ```docker-compose.yaml``` file\n",
    "    * To test only the kinesis part we can use ```docker-compose up kinesis```\n",
    "    * \"Use AWS locally\": ```aws endpoind-url=http://localhost:4566 kinesis list-streams```\n",
    "    * Create a stream (locally): ```aws endpoind-url=http://localhost:4566 kinesis create stream --stream-name <value> [--shard-count-value <value>]```, i.e. ```aws endpoind-url=http://localhost:4566 kinesis create stream --stream-name ride-predictions --shard-count-value 1```\n",
    "    * In our docker-compose.yaml, we specify this by the variable ```KINESIS_ENDPOINT_URL=http://kinesis:4566``` to configure our code to go to localstack istead to aws\n",
    "    * We also have to add this to our ```model.py``` script. This is done by the function ```create_kinesis_client```\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c15cc7ff",
   "metadata": {},
   "source": [
    "## Code Quality: Linting and Formatting\n",
    "\n",
    "* PEP8 - Stype Guide for Python\n",
    "* linters help to see, whether a code follows this guide, e.g. ```pylint```\n",
    "* \"Pylint is a static code analysis tool for the Python programming language.\"\n",
    "* ```pipenv install --dev pylint```\n",
    "* Use this for specific files, e.g. ```pylint model.py``` or to an entire folder ```pylint --recursive=y .``` in the terminal\n",
    "* Or more convinient in Visual Studio: ```View > Command Palette > Python: Select Linter > pylint```, run it: ```View > Command Palette > run linting```, we see then all suggestions as underlined code\n",
    "* We can configure what kind of suggestions should be shown. We can create a file ```.pylintrc``` and e.g. add what kind of suggestions should be ignored.\n",
    "* Alternative to ```pylint```: Many packages (including ```pylint```) use a configuration file called ```pyproject.toml```. Create this file and move the content from ```.pylintrc``` there.\n",
    "* You can also disable locally some warinings, e.g.: \n",
    "```\n",
    "def lambda_handler(event, context):\n",
    "    # pylint: disable=unused-argument\n",
    "    return model_service.lamda_handler(event)\n",
    "```\n",
    "* Now we use the packages ```black``` for formatting and ```isort``` for sorting the imports\n",
    "* ```black --diff . | less```, use ```black --skip-string-normalization --diff . |less``` to ignore single quote issues\n",
    "* We put this into the ```pyproject.toml``` file\n",
    "* apply the changes ```black .```\n",
    "* similar use ```isort --diff . | less```\n",
    "* We can do this all automatically:\n",
    "```\n",
    "isort .\n",
    "black .\n",
    "pylint --recursive=y .\n",
    "pytest tests/\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "05d897f1",
   "metadata": {},
   "source": [
    "## Pre-commit Hooks\n",
    "* To make sure, the tests we want to do are really done, we can make them always before we commit something to git.\n",
    "* Use git ```pre-commit hook```: ```pip install pre-commit```\n",
    "* We use our virtual environment: ```cd code```, ```pipenv shell```, ```pipenv install --dev pre-commit```\n",
    "* This allows us to define pre-commit hooks\n",
    "* When we go to the base folder of our repository, we have a folder called ```.git```, in this folder is a folder called ```hooks```. This contains a file called ```pre-commit.sample```\n",
    "* We only want to run pre-commit hooks for the folder ```code``` in our repo. \n",
    "* run ```git init``` in this folder to pretend this is a git repo (delete after the lecture)\n",
    "* run ```pre-commit```, we need to create a config file ```.pre-commit.yaml```\n",
    "* create the sample comfig with ```pre-commit sample-config > .pre-commit.yaml```\n",
    "* ```pre-commit install``` creates a hook at ```.git/hooks/pre-commit```\n",
    "* When we clone a repo, we need to install the pre-commit hooks!\n",
    "* When we now commit changes, the pre-commit hooks are run\n",
    "* We now need to add the hooks we want to run to the ```.pre-commit.yaml``` file"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "efb17f31",
   "metadata": {},
   "source": [
    "## Makefiles and Make\n",
    "* create a new file called ```Makefile``` in the ```code``` folder\n",
    "* when we run ```make run``` this Makefile will be executed.\n",
    "* Simple example:\n",
    "```\n",
    "test:\n",
    "\techo test\n",
    "run: test # means that run depends on test\n",
    "\techo run\n",
    "```\n",
    "when we execute ```make run``` we get this output:\n",
    "```\n",
    "echo test\n",
    "test\n",
    "echo run\n",
    "run\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1015dc29",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ml-zoomcamp",
   "language": "python",
   "name": "ml-zoomcamp"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
