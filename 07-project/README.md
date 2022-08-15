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

## Repository Structure

**data-exploration.ipynb:** Notebook containing data exploration

**training.py:** Data preparation and Model training including experiment tracking and model registry using mlflow
* Model used: XGBoost for Regression
* Environment
	* The needed packages are saved in project-env.yml and can be converted into a conda environment using ```conda env create --name project-env --file=project-env.yml```
	* Activate the environment with ```conda activate project-env```
* Start the Server
	* Experiment tracking and model registry UI can be accessed via ```mlflow server --backend-store-uri sqlite:///mlruns.db  --default-artifact-root artifacts```
* Trainine
	* Run the training script with: ```python3 training.py --input-data <path/to/input-data.csv> --output <path/to/output>``` (and optional other parameters)
* Hyperparameter tuning
	* Hyperparameter tuning done via Optuna
	* Number of trials for hyperparameter tuning can be changed using the parameter ```n-trials```, default value is set to 200, e.g. ``python3 training.py --n-trials 50```
	* Model parameters for hyperparameter tuning can also be change via the command line, e.g. ```e-estimators```, ```max-depth```, ```gamma```, ```eta```, etc.
* Mlflow experiment tracking and model registry
	* mlflow tracking server: yes, local server
	* mlflow backend store: sqlite database
	* mlflow artifacts store: local filesystem
	* Best model is registered using mlflow 
	* Experiment tracking and model registry UI can be accessed via ```localhost:5000``` in the browser
* Orchestration using prefect
	* The ```main``` function is turned into a prefect ```flow```
	* The functions ```read_data```, ```normalize```, ```onehot```, and ```training``` are turned into tasks
	* To the the prefect UI use ```prefect orion``` and browse to ```localhost:4200```

