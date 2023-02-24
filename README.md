[![PyPI version](https://badge.fury.io/py/bluemist.svg)](https://badge.fury.io/py/bluemist)
[![Documentation Status](https://readthedocs.org/projects/bluemist-ai/badge/?version=latest)](https://bluemist-ai.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/shashanka89/bluemist-ai)

# Bluemist AI

Bluemist AI is a low code machine learning library written in Python to develop, evaluate and deploy automated ML
pipleines. 

It acts as a wrapper service on top of sklearn, numpy, pandas, mlflow and FastAPI.

Visualization are created using pandas-profiling, sweetviz, dtale and autoviz. 

## Features
- Native integration for data extraction with MySQL, PostgreSQL, MS SQL, Oracle, MariaDB, Amazon Aurora and Amazon S3
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Trains data across multiple algorithms and provide comparison metrics
- Hyperparameter tuning
- Experiment tracking
- API deployment

For more detail please visit @ https://www.bluemist-ai.one

## User installation

### Method 1:

```{python}
source bluemist-env/bin/activate
pip install -U bluemist
```

### Method 2 (recommended):
It is advised to setup a separate python environment to avoid conflicts with package dependencies. 
This can be done as follows :

- Install the package ``virtualenv``

```{python}
pip install virtualenv
```

- Create a separate directory where bluemist environment will be created
- 
```{python}
mkdir /path/to/bluemist-ai
cd /path/to/bluemist-ai
```

- Create the bluemist environment
- 
```{python}
virtualenv bluemist-env
```

- Activate the environment and install bluemist
- 
```{python}
source bluemist-env/bin/activate
pip install -U bluemist
```

### Method 3:

bluemist package can be installed using ``pipx`` utility. It automatically creates an isolated environment to run the
bluemist package

```{python}
pip install pipx
pipx install bluemist
pipx upgrade bluemist
```


## License

Bluemist AI source code is licensed under the MIT License

See [Third Party Libraries](https://github.com/mist-projects/bluemist-ai/wiki/Third-Part-Libraries) for license details of third party libraries included in the distribution.
