# ARISA-MLOps Heart Disease Classification

## Project description

The aim of this project is to build, evaluate, and deploy a machine learning model that can predict the presence of heart disease based on patient health data. Using a dataset from Kaggle, we aim to support early diagnosis and risk assessment for cardiovascular conditions.

## Model Risk Assessment

| **Risk Category**        | **Description**                                                                 | **Mitigation Strategy**                                                                 |
|--------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Data Bias**            | Lack of demographic features (e.g., ethnicity, geography) may reduce fairness and generalization across populations. | Clearly document dataset limitations; consider adding diverse features in future versions. |
| **Model Dependency**     | Strong reliance on CatBoost may limit portability and increase maintenance risk if the library becomes unsupported. | Track library versions; consider exporting to ONNX or evaluating alternative models.     |
| **Overfitting**          | Model might perform well on training data but poorly on new data.              | Use early stopping, cross-validation, and monitor validation metrics closely.            |
| **Reproducibility**      | Results may vary if experiments aren't properly tracked or versioned.          | Ensure all experiments use `random_state`, `joblib`, `mlflow`, and `git` version control. |


## Dataset

Dataset: Heart Disease Dataset (https://www.kaggle.com/datasets/mexwell/heart-disease-dataset)<br>
Author: mexwell<br>
Subject Area: Heart Conditions, Drugs and Medications, Binary Classification, Medicine<br>

The heart disease dataset is curated by combining 5 popular heart disease datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:
- Cleveland
- Hungarian
- Switzerland
- Long Beach VA
- Statlog (Heart) Data Set

This dataset consists of 1190 instances with 11 features. These datasets were collected and combined at one place to help advance research on CAD-related machine learning and data mining algorithms, and hopefully to ultimately advance clinical diagnosis and early treatment.

| No   | Attribute                              | Code given          | Unit                    | Data type |
|------|----------------------------------------|---------------------|-------------------------|-----------|
| 1    | Age                                    | Age                 | in years                | Numeric   |
| 2    | Sex                                    | Sex                 | 1, 0                    | Binary    |
| 3    | Chest pain type                        | Chest pain type     | 1, 2, 3, 4              | Nominal   |
| 4    | Resting blood pressure                 | Resting bp s        | in mm Hg                | Numeric   |
| 5    | Serum cholesterol                      | Cholesterol         | in mg/dl                | Numeric   |
| 6    | Fasting blood sugar                    | Fasting blood sugar | 1, 0 > 120 mg/dl        | Binary    |
| 7    | Resting electrocardiogram results      | Resting ecg         | 0, 1, 2                 | Nominal   |
| 8    | Maximum heart rate achieved            | Max heart rate      | 71–202                  | Numeric   |
| 9    | Exercise induced angina                | Exercise angina     | 0, 1                    | Binary    |
| 10   | Oldpeak = ST                           | Oldpeak             | Depression              | Numeric   |
| 11   | The slope of the peak exercise ST seg. | ST slope            | 0, 1, 2                 | Nominal   |
| 12   | Class (Heart disease presence)         | Target              | 0, 1                    | Binary    |


## ML Model Description

The classifier is implemented using CatBoostClassifier, a gradient boosting algorithm on decision trees optimized for categorical and numerical tabular data. A core feature of CatBoost is its native support for categorical variables, which are internally transformed using target-based encoding schemes derived from order statistics. This approach avoids the need for explicit one-hot or label encoding and preserves the statistical distribution of features. <br>

CatBoost mitigates the risk of overfitting through the use of ordered boosting, a technique that constructs training sets for each iteration in a way that prevents target leakage. This is achieved by ensuring that the target value of an observation is never used in its own encoding. Additionally, the algorithm includes regularization techniques and built-in cross-validation to enhance generalization.<br>

CatBoost provides competitive performance on structured datasets, maintains consistency across random seeds, and is efficient in terms of training speed and handling of missing values.


### Hyperparameter tuning:
- Implemented using Optun and mlflow.start_run(nested=True)
- Parameters are saved to the best_params.pkl file and logged in MLflow

### Cross-validation
- Implemented by catboost.cv with 5-fold stratified shuffle
- Results (F1 and logloss) are visualized with standard errors using Plotly

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ARISA_DSML and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ARISA_DSML   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ARISA_DSML a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Setup

### Prerequisites

- Git & GitHub — version control and collaboration
- Python 3.11.9 or higher
- Jupyter Notebook environment
- Pandas and NumPy libraries
- Scikit-learn library
- Matplotlib and Seaborn library
- MLflow — experiment tracking

### Running on local machine

Install python 3.11 with py manager on your local machine.<br>
Install Visual Studio Code on your local machine.<br>
Create a kaggle account and create a kaggle api key kaggle.json file.<br>
Move the kaggle.json to<br>
C:\Users\USERNAME\.kaggle folder for windows,<br>
/home/username/.config/kaggle folder for mac or linux.<br>


Clone the repository to your local machine:
``` bash
git clone https://github.com/wojciech-karwowski/ARISA-MLOps-Assignment.git
```

Change the working directory:
``` bash
cd ARISA-MLOps-Assignment
```

Create virtual environment:
``` bash
py -3.11 -m venv .venv
```

Activate virtual environment:
``` bash
# on Windows:
.\.venv\Scripts\activate

# on Linux/Mac:
source .venv/bin/activate
```

Install dependencies:
``` bash
pip install -r requirements.txt
```

Experiment tracking in MLFlow:
``` bash
mlflow ui
```
then open http://localhost:5000 in your browser to view logged experiments.

## Traceability and Reproducibility

- The entire infrastructure is managed as code (IaC) and maintained in the Git repository under .github/workflows.
- All changes are introduced through pull requests, triggering automated deployment via GitHub Actions.
- Direct commits to the main branch are protected by workflow.
- CI/CD pipelines are used for applying changes, ensuring consistency and traceability.
- Source code for data preprocessing, model training, and API logic is stored under a version-controlled ARISA_DSML/ directory.
- All modifications to the codebase are implemented via pull requests to ensure auditability and peer review.
- The project environment, including virtual environment, dependencies, and configuration, is fully reproducible using pyproject.toml, setup.cfg, and optionally venv.
- A unified runtime ensures that the code runs identically in local development and GitHub Actions without adjustments.






- Introducing changes only via pull requests, with automatic deployment via GitHub Actions
- Securing the main branch from direct commits
- Pipeline CI/CD as the only source of implementing changes
- Configuring two environments according to the assumptions of the repository and pipelines
- Access of environments to identical data (e.g. consistency of input data in pipelines)

- Data processing, model training and API code saved in versioned folder ARISA_DSML/
- All code changes introduced by PRs
- Project environment (venv + pyproject.toml, setup.cfg files) fully reproducible – defined as code
- Unified runtime environment (works locally and on GitHub Actions without modifications)
- Possibility to clearly link model launch to:
  - source code (commit Git),
  - infrastructure (pipeline, environment),
  - training data (CSV file or artifact in MLflow).

## Code Quality

The CI pipeline implemented via GitHub Actions automatically validates configuration and runs tests whenever code changes are introduced.

All updates to the codebase must go through pull requests, and are subject to review by other team members before being merged into the main branch.

Pre-commit hooks using flake8 enforce code style consistency across the project.

The ML codebase is modular and includes unit tests for:
 - Data processing
 - Model training
 - Predictions

After each pull request, the CI pipeline (lint-code.yml) runs linting and validation checks.

A dedicated prediction pipeline (predict_on_model_change.yml) is triggered when change of:
- test.csv, train.csv
- predict.py
- catboost_model_diabetes.cbm

A separate retraining pipeline (retrain_on_change.yml) is executed when moddifed are:
- test.csv, train.csv
- preproc.py, train.py
- best_params.pkl

Documentation is treated as code and maintained in Markdown files and Python docstrings.

Each model update is accompanied by release notes in pull request descriptions.


## Monitoring & Support

- An automated alerting mechanism detects issues such as pipeline failures, deployment errors, or execution timeouts, enabling rapid response and resolution.

- Each execution of the CI/CD pipeline is fully logged, including runtime duration, status (success/failure), and detailed console output, ensuring traceability and auditability.

-  every model training cycle, key validation metrics (e.g., F1-score, Accuracy, LogLoss) are logged and versioned. Historical performance is tracked using MLflow, facilitating model evaluation over time.

- Feature distributions (e.g., age, cholesterol, resting blood pressure) are monitored for potential data drift. These distributions are regularly compared with training data, and if significant deviation is detected, alerts are issued or automatic retraining is triggered.