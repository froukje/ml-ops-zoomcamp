# Project: Energy Efficiency

Analyzing the energy efficiency of buildings gains more and more importancy considering climate change and rising energy costs. In this project I will analyse and predict the heat load of a building given different building characteristics.

## Objective

Predict the "Heating Load", depending on different building features. 

Note: The Dataset contains another predictable variable, "Cooling Load", which can also be predicted from the given features. However, in this project we will focus on only one model predicting the "Heating Load".

## Dataset
Data source: https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset

The dataset was created by Angeliki Xifara (angxifara '@' gmail.com, Civil/Structural Engineer) and was processed by Athanasios Tsanas (tsanasthanasis '@' gmail.com, Oxford Centre for Industrial and Applied Mathematics, University of Oxford, UK).

### Data Set Information:

Energy analysis using 12 different building shapes simulated in Ecotect. The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters. We simulate various settings as functions of the afore-mentioned characteristics to obtain 768 building shapes. The dataset comprises 768 samples and 8 features, aiming to predict two real valued responses. It can also be used as a multi-class classification problem if the response is rounded to the nearest integer.

### Attribute Information:

The dataset contains eight attributes (or features, denoted by X1…X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.

Specifically:
* X1 Relative Compactnes
	* This is the volume to surface ratio. Buildings with a lower compactness have a larger surface area for a given volume.
* X2 Surface Area ($m²$)
* X3 Wall Area ($𝑚²$)
* X4 Roof Area ($𝑚²$)
* X5 Overall Height ($m$)
* X6 Orientation (2: North, 3: East, 4: South, 5: West)
* X7 Glazing Area (0%, 10%, 25%, 40% (of floor area))
	* This is the area of transparent material, not including the window frame.
* X8 Glazing Area Distribution ((Variance) - 1: Uniform, 2: North, 3: East, 4: South, 5: West)
* Y1 Heating Load ($𝑘Wh/m²$)
* (Y2 Cooling Load ($kWh/m²$))

### Relevant Papers:

A. Tsanas, A. Xifara: 'Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools', Energy and Buildings, Vol. 49, pp. 560-567, 2012

### For further details on the data analysis methodology:

A. Tsanas, 'Accurate telemonitoring of Parkinsonâs disease symptom severity using nonlinear speech signal processing and statistical machine learning', D.Phil. thesis, University of Oxford, 2012

## Quickstart

To run the web-app the follwing steps are necessary:
* Train and save model
	* create environment
	* activate environment
	* prefect deployment create prefect-deploy.py (or without deployment ...)
	* the best model is saved in model reistry
* Run web app
	* create docker image
	* run docker-compose up

## Repository Structure

**```data-exploration.ipynb```:** Notebook containing data exploration

**Note:** The individual steps of the project are stored in individual files, to be able to run the tasks individually (e.g. do only  model training without orchestration). This is only done for the project to see how my working steps were and not necessary.

**```prefect-deploy-py```:** Data preparation, Model training and Deployment:
* Model used: XGBoost for Regression
* Environment:
	* The needed packages are saved in project-env.yml and can be converted into a conda environment using ```conda env create --name project-env --file=project-env.yml```
	* Activate the environment with ```conda activate project-env```
* Start the Server
	* Start server for tracking and model registry ```mlflow server --backend-store-uri sqlite:///mlruns.db  --default-artifact-root artifacts```
* Training
	* To run only training with experiment tracking and model registry use ```exp-tracking.py``` and run it with: ```python3 exp-tracking.py --input-data <path/to/input-data.csv> --output <path/to/output>``` (and optional other parameters)
* Hyperparameter tuning
	* Hyperparameter tuning is done via Optuna
	* Number of trials for hyperparameter tuning can be changed using the parameter ```n-trials```, default value is set to 200, e.g. ```python3 exp-tracking.py --n-trials 50```, to change it for the final ```training.py``` file, you need to change it directly in the script.
	* Model parameters for hyperparameter tuning can also be change via the command line, e.g. ```n-estimators```, ```max-depth```, ```gamma```, ```eta```, etc. for ```exp-tracking.py```. For the final script, they need to be changed in the script.
* Mlflow experiment tracking and model registry (Following videos from week 2)
	* mlflow tracking server: sqlite database
	* mlflow backend store: sqlite database
	* mlflow artifacts store: local filesystem
	* Best model (lowest validation metric: RMSE) is automatially registered using mlflow. Note: Even better would be to compare with the previous registered model and only register the new one if it is better than the previous one. 
	* Experiment tracking and model registry UI can be accessed via ```localhost:5000``` in the browser
* Orchestration using prefect (Following videos from week 3)
	* The ```main``` function is turned into a prefect ```flow```
	* The functions ```read_data```, ```normalize```, ```onehot```, and ```training``` are turned into tasks
	* To the the prefect UI use ```prefect orion start``` and browse to ```localhost:4200```
	* To start a prefect flow (without deployment) use the script ```prefect-flow.py``` and run it as before the ```exp-tracking.py``` file
	* A deployment is used to run the script every 5 minutes. 
	* To run the prefect deployment use ```prefect deployment create prefect-deploy.py```
	* Note: to create the deployment, I had to change the code slightly, as the argparse is not working (and also not useful), when the flow is scheduled. 
	* Create a work queue in the UI, as shown in video 3.5 of the course
	* Spin up an agend ```prefect agent start <workqueue-id>```, e.g. ```prefect agent start a4bdb288-7329-4a1c-992f-fe62cd898af9```

**```predict.py```:** Deploy model as a web service
* The model is deployed as a web service (Following videos from week 4)
	* The model and other needed artifacts are loaded from mlflow model registry. Note: This would be easier following the advice from the videos of creating a pipeline and saving everything together, however, I decided to leave the code as it is to have more functions and with that more flows for the purpose of getting to know the workflow of orchestration better.
	* The model version needs to be set manually in the ```predict.py``` script!
	* A virtual envirenment using ```Pipenv``` is created containing ```flask```, ```xgboost==1.6.1```, ```scikit-learn```, ```gunicorn```, ```requests```, ```mlflow``` are created
	* Activate the envirenment by ```pipenv shell```
	* The ```predict.py``` file can be tested locally using ```test-predict.py```. This gives the prediction of one specific input example
	* The way this is implemented the predict.py file depends on that the tracking server is running. As mentioned in the videos, ideally this dependency should be removed. However, I wanted to try to automatically get the registered model, without putting manually the run_id.
	* The flask app can be tested locally by starting ```gunicorn --bind=0.0.0.0:9696 predict:app``` and then run ```test-predict-flask.py``` in another terminal. This should give the same result as ```test-predit.py```
**```predict-docker.py```:** Deploy model as a web service using docker	
* The app can be started in Docker
	* Build the docker image: ```docker build -t heat-loading-service:v1 .```
	* Run the container: ```docker-compose up```
* Note: for running the app in docker I changed the code from ```predict.py``` and made it independend of the tracking server. However, now manually the ```RUN_ID``` of the selected model has to be given in the script.

TODO:
* check port 5000
* check if outout
* conda env
