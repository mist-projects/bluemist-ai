# Bluemist AI

[![Generic badge](https://img.shields.io/badge/python-3.9-blue.svg)](https://shields.io/)
[![PyPI version](https://badge.fury.io/py/bluemist.svg)](https://badge.fury.io/py/bluemist)
![GitHub](https://img.shields.io/github/license/mist-projects/bluemist-ai)
[![Documentation Status](https://readthedocs.org/projects/bluemist-ai/badge/?version=latest)](https://bluemist-ai.readthedocs.io/en/latest/?badge=latest)

Bluemist AI is a low code machine learning library written in Python to develop, evaluate and deploy automated Machine
Bluemist AI is an open source, low code machine learning library written in Python to develop, evaluate and deploy automated Machine
Learning models and pipleines.

It acts as a wrapper service on top of sklearn, numpy, pandas, mlflow and FastAPI. Visualization are created using
pandas-profiling, sweetviz, dtale and autoviz. 

## Features
- Native integration for data extraction with MySQL, PostgreSQL, MS SQL, Oracle, MariaDB, Amazon Aurora and Amazon S3
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Trains data across multiple algorithms and provide comparison metrics
- Hyperparameter tuning
- Experiment tracking
- API deployment

For the detailed list of supported and upcoming features, visit https://www.bluemist-ai.one

Full documentation is available @ https://bluemist-ai.readthedocs.io

## User installation

#### Method 1

To install minimal version of the package with hard dependencies listed in  [requirements.txt](https://github.com/mist-projects/bluemist-ai/blob/f0f6f74e70b24171a6df13b90220139ede70f4e3/requirements.txt)

```{python}
pip install -U bluemist
```

To install the complete package including optional dependencies listed
in  [requirements-optional.txt](https://github.com/mist-projects/bluemist-ai/blob/8db75fc52824783f7d9e9a2ad2f65e51d3a30e33/requirements-optional.txt).
Refer [Minimal package vs Full package](#minimal-package-vs-full-package) for more details.

```{python}
pip install -U bluemist[complete]
```

#### Method 2 (recommended)
It is advised to setup a separate python environment to avoid conflicts with package dependencies. 
This can be done as follows :

- Install the package ``virtualenv``

```{python}
pip install virtualenv
```

- Create a separate directory where bluemist environment will be created

```{python}
mkdir /path/to/bluemist-ai
cd /path/to/bluemist-ai
```

- Create the bluemist environment

```{python}
virtualenv bluemist-env
```

- Activate the environment and install bluemist

```{python}
source bluemist-env/bin/activate
pip install -U bluemist
```

#### Method 3

bluemist package can be installed using ``pipx`` utility. It automatically creates an isolated environment to run the
bluemist package

```{python}
pip install pipx
pipx install bluemist
pipx upgrade bluemist
```

## Minimal package vs Full package
Below functionalities are available only with complete package installation

- Data extraction from RDBMS or cloud
- EDA Visualizations using dtale and sweetviz

Alternatively a single optional package can be installed using the ``pip`` command. For example, if you would like to
extract data from Amazon S3 but do not wish to install other optional packages :

```{python}
pip install boto3
```

## License

Bluemist AI source code is licensed under the MIT License

See [Third Party Libraries](https://github.com/mist-projects/bluemist-ai/wiki/Third-Part-Libraries) for license details of third party libraries included in the distribution.

