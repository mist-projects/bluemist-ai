# Bluemist AI

[![Generic badge](https://img.shields.io/badge/python-3.9-blue.svg)](https://shields.io/)
[![PyPI version](https://badge.fury.io/py/bluemist.svg)](https://badge.fury.io/py/bluemist)
![GitHub](https://img.shields.io/github/license/mist-projects/bluemist-ai)
[![Documentation Status](https://readthedocs.org/projects/bluemist-ai/badge/?version=latest)](https://bluemist-ai.readthedocs.io/en/latest/?badge=latest)

Bluemist AI is an advanced, open-source, low-code machine learning library developed in Python. Its primary purpose is
to facilitate the development, evaluation, and deployment of automated machine learning models and pipelines.

The library acts as a convenient wrapper service, integrating seamlessly with popular Python libraries such as
scikit-learn, NumPy, pandas, mlflow, and FastAPI. For visualization purposes, Bluemist AI leverages the capabilities of
pandas-profiling, sweetviz, dtale, and autoviz.

## Key Features

Bluemist AI offers a range of powerful features, including:

- **Native Data Integration**: Seamlessly extract data from various sources, including MySQL, PostgreSQL, MS SQL, Oracle, MariaDB, Amazon Aurora, and Amazon S3, providing a unified interface for working with diverse datasets.
- **Exploratory Data Analysis (EDA)**: Conduct thorough EDA on your data, enabling comprehensive insights into the characteristics, distributions, and relationships within the dataset.
- **Data Preprocessing**: Streamline the preprocessing pipeline by leveraging Bluemist AI's built-in preprocessing capabilities, allowing you to handle missing data, perform feature scaling, encoding, and other essential preprocessing tasks efficiently.
- **Algorithm Selection and Comparison**: Bluemist AI facilitates training data across multiple algorithms, enabling you to compare and evaluate various models using comprehensive metrics and insights.
- **Hyperparameter Tuning**: Optimize model performance with built-in hyperparameter tuning functionalities, automatically searching for the optimal combination of hyperparameters to achieve the best results.
- **Experiment Tracking**: Keep track of your experiments effortlessly, storing essential metadata, parameters, and results, allowing for easy reproducibility and experimentation management.
- **API Deployment**: Bluemist AI simplifies the process of deploying machine learning models as APIs, making it easier to integrate your models into production systems and applications.

These features make Bluemist AI a powerful tool for developing, evaluating, and deploying machine learning models efficiently and effectively.

## Getting Started

For the detailed list of supported and upcoming features, visit https://www.bluemist-ai.one

Full documentation is available @ https://bluemist-ai.readthedocs.io

## User installation

#### Method 1

To install minimal version of the package with hard dependencies listed in  [requirements.txt](https://github.com/mist-projects/bluemist-ai/blob/f0f6f74e70b24171a6df13b90220139ede70f4e3/requirements.txt)

```{python}
pip install -U bluemist
```

To install the complete package including optional dependencies listed in  [requirements-optional.txt](https://github.com/mist-projects/bluemist-ai/blob/8db75fc52824783f7d9e9a2ad2f65e51d3a30e33/requirements-optional.txt). Refer [Minimal package vs Full package](#minimal-package-vs-full-package) for more details.

```{python}
pip install -U bluemist[complete]
```

#### Method 2 (recommended)
It is advised to set up a separate Python environment to avoid conflicts with package dependencies. Follow the steps below:
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

## Support for GPU/CPU Acceleration

#### GPU Acceleration

- If your system is equipped with an NVIDIA GPU and you want to utilize GPU acceleration for model training, you can
  install the RAPIDS cuML package by executing the following command:

```{python}
pip install cudf-cu11 cuml-cu11 --extra-index-url=https://pypi.nvidia.com
```

- The `cuml` package provides GPU-accelerated implementations of various machine learning algorithms. By installing this
  package, you can leverage the power of your GPU to significantly improve the speed and performance of your machine
  learning workflows.

- For the list of supported algorithms, please
  refer https://docs.rapids.ai/api/cuml/stable/api/#regression-and-classification

- Acceleration extensions can ben enabled by passing the parameter ``enable_acceleration_extensions=True`` during
  the ``initialize`` phase.

#### CPU Acceleration

- If your system is equipped with an Intel CPU and you want to utilize CPU acceleration for model training, you can
  install the Intel® Extension for Scikit-learn package by executing the following command:

```{python}
pip install -U scikit-learn-intelex
```

- The `scikit-learn-intelex` package provides CPU-accelerated implementations of Scikit-learn algorithms. By installing
  this package, you can leverage the power of your Intel CPU to significantly improve the speed and performance of your
  machine learning workflows.

- For the list of supported algorithms, please refer to
  the [Intel® Extension for Scikit-learn documentation](https://intel.github.io/scikit-learn-intelex/algorithms.html#on-cpu).

- Acceleration extensions can ben enabled by passing the parameter ``enable_acceleration_extensions=True`` during
  the ``initialize`` phase.

## Minimal package vs Full package
Below functionalities are available only with complete package installation

- Data extraction from RDBMS or cloud
- EDA Visualizations using dtale and sweetviz
- CPU/GPU Acceleration

Alternatively a single optional package can be installed using the ``pip`` command. For example, if you would like to
extract data from Amazon S3 but do not wish to install other optional packages :

```{python}
pip install boto3
```

## License

This project is licensed under the [MIT License](LICENSE).

See [Third Party Libraries](https://github.com/mist-projects/bluemist-ai/wiki/Third-Part-Libraries) for license details of third party libraries included in the distribution.

