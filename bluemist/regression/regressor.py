"""
Performs model training, testing, evaluations and deployment
"""

# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.2
# Email: dew@bluemist-ai.one
# Created:  Jun 22, 2022
# Last modified: June 19, 2023

import importlib
import logging
import os
import time
from logging import config
import pandas as pd

from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import bluemist
from bluemist import environment
from bluemist.pipeline.bluemist_pipeline import add_pipeline_step, save_model_pipeline, clear_all_model_pipelines
from bluemist.preprocessing import preprocessor
from bluemist.regression.constant import multi_output_regressors, multi_task_regressors, unsupported_regressors, \
    base_estimator_regressors
from bluemist.utils.constants import GPU_BRAND_INTEL, GPU_BRAND_NVIDIA, GPU_ACCELERATION_NVIDIA, CPU_ACCELERATION_INTEL, \
    CPU_BRAND_INTEL

from bluemist.utils.metrics import metric_scorer
from sklearn.utils import all_estimators

from bluemist.utils.scaler import get_scaler
from bluemist.utils import generate_api as generate_api
from bluemist.artifacts.api import predict

from IPython.display import display, HTML

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def import_mlflow():
    import mlflow
    return mlflow


def measure_execution_time(start_time):
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Convert elapsed time to seconds, minutes, milliseconds, and hours
    milliseconds = int(elapsed_time * 1000) % 1000
    seconds = elapsed_time % 60
    minutes = (elapsed_time // 60) % 60
    hours = (elapsed_time // 3600)

    execution_time = "{:02d}:{:02d}:{:02d},{:03d}".format(int(hours), int(minutes), int(seconds), int(milliseconds))
    return execution_time


def initialize_mlflow(mlflow_experiment_name):
    mlflow = import_mlflow()

    logger.info('Initializing MLFlow...')
    if mlflow_experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        if not experiment:
            mlflow.create_experiment(name=mlflow_experiment_name,
                                     artifact_location=BLUEMIST_PATH + '/' + 'artifacts/experiments/mlflow')

        if experiment is not None and experiment.lifecycle_stage == 'deleted':
            logger.info('Restoring MLFlow experiment :: {}'.format(mlflow_experiment_name))
            client = mlflow.tracking.MlflowClient()
            client.restore_experiment(experiment.experiment_id)

        mlflow.set_experiment(mlflow_experiment_name)


def get_estimators(multi_output=False, multi_task=False, names_only=True):
    """
        **Returns the list of available regression estimators**

        multi_output : bool, default=False
            Future use
        multi_task : bool, default=False
            Future use
        names_only : bool, default=True
            Returns only the estimator name without metadata
    """

    estimators = all_estimators(type_filter='regressor')
    logger.debug('All available estimators :: {}'.format(estimators))

    estimators_to_remove = []
    for estimator in estimators:
        if not multi_output and estimator[0] in multi_output_regressors:
            estimators_to_remove.append(estimator)
        if not multi_task and estimator[0] in multi_task_regressors:
            estimators_to_remove.append(estimator)
        if unsupported_regressors and estimator[0] in unsupported_regressors:
            estimators_to_remove.append(estimator)
        if base_estimator_regressors and estimator[0] in base_estimator_regressors:
            estimators_to_remove.append(estimator)

    logger.debug('Estimators not supported by Bluemist AI :: {}'.format(estimators_to_remove))

    for estimator_to_remove in estimators_to_remove:
        estimators.remove(estimator_to_remove)

    if bool(names_only):
        return [estimator[0] for estimator in estimators]

    logger.info('Estimators available for modelling :: {}'.format(estimators))
    return estimators


def deploy_model(estimator_name, host='localhost', port=8000):
    """
        estimator_name : str
            Name of the estimator to be deployed
        host : {str, IPv4 or IPv6}, default='localhost'
            Hostname or IP address of the machine where the API will be deployed
        port : int, default=8000
            API listening port
    """

    logger.info('Generating API code to deploy the model :: {}'.format(estimator_name))
    generate_api.generate_api_code(estimator_name=estimator_name,
                                   initial_column_metadata=preprocessor.initial_column_metadata_for_deployment,
                                   encoded_column_metadata=preprocessor.encoded_columns_for_deployment,
                                   target_variable=preprocessor.target_for_deployment)
    importlib.reload(predict)
    logger.info('Starting API server on host {} and port {}'.format(host, port))
    predict.start_api_server(host=host, port=port)


def train_test_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        tune_models=None,
        metrics='default',
        multi_output=False,
        multi_task=False,
        target_scaling_strategy=None,
        save_pipeline_to_disk=True,
        experiment_name=None,
        run_name=None):
    """
        **Trains the data on the given dataset, evaluate the models and returns comparison metrics**

        X_train : pandas dataframe
            Training data
        X_test : pandas dataframe
            Test data
        y_train : array of shape (X_train.shape[0],)
            Target values of training dataset
        y_test : array of shape (X_test.shape[0],)
            Target values of test dataset
        tune_models : {'all', None} or list of models to be trained, default=None
            all: tune all regression models
            list: list of models to be trained
            None: hyperparameter tuning will not be performed
        metrics : {'all', 'default'}, default='default'
            - all:
                mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error,
                mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error, mean_poisson_deviance,
                mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss
            - default:
                mean_absolute_error, mean_squared_error, r2_score
        multi_output : bool, default=False
            Future use
        multi_task : bool, default=False
            Future use
        target_scaling_strategy : {'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', None}, default=None
            Scales the target variable before training the model
        save_pipeline_to_disk : bool, default=True
            Save preprocessor and model training pipeline to the disk. Should be set to True if needs model to be deployed
            as an API
        experiment_name : str, default=None
            Name of the experiment
        run_name : str,default=None
            Name of the run within the experiment

        Examples
        ---------
        *Regression*

        .. raw:: html
            :file: ../../code_samples/quickstarts/regression/regression_hyperparameter_tuning.html

    """

    tune_all_models = False
    models_to_tune = []
    target_scaler = None
    sklearnex_algorithms = None

    if target_scaling_strategy is not None:
        target_scaler = get_scaler(target_scaling_strategy)

    if isinstance(tune_models, str) and tune_models == 'all':
        tune_all_models = True
    elif isinstance(tune_models, list):
        models_to_tune = tune_models

    if experiment_name is not None:
        initialize_mlflow(experiment_name)

    # Get patch names from sklearnex
    if environment.available_cpu == CPU_BRAND_INTEL:
        from sklearnex import sklearn_is_patched, get_patch_names
        sklearnex_algorithms = get_patch_names()

    df = pd.DataFrame()

    estimators = get_estimators(multi_output, multi_task, names_only=False)
    clear_all_model_pipelines()

    counter = 0

    # If hyperparameter tuning is requested for specific models, limit the overall training to those models to save time
    if tune_models is not None and not tune_all_models and models_to_tune:
        estimators_to_skip = []
        for estimator in estimators:
            if estimator[0] not in models_to_tune:
                estimators_to_skip.append(estimator)

        for estimator_to_skip in estimators_to_skip:
            estimators.remove(estimator_to_skip)

    for estimator_name, estimator_class in (pbar := tqdm(estimators, colour='blue')):
        start_time = time.time()
        pbar.set_description(f"Training {estimator_name}")

        counter = counter + 1
        estimator_execution_device = 'CPU'

        # No tuning OR Tune All OR Tune Few
        if tune_models is None or tune_all_models or estimator_name in models_to_tune:
            logger.info('############  Regressor in progress :: {} ############'.format(estimator_name))
            regressor = None

            try:
                if environment.available_gpu == GPU_BRAND_NVIDIA:
                    import cuml
                    cuml.set_global_output_type('array')  # cuML will return predictions of type cupy.ndarray
                    if hasattr(cuml, estimator_name):
                        regressor = getattr(cuml, estimator_name)()
                        estimator_execution_device = GPU_ACCELERATION_NVIDIA
                        logger.info('Regressor class from cuML :: {}'.format(regressor.__class__.__module__ + '.' + regressor.__class__.__name__))

                if regressor is None and environment.available_cpu == CPU_BRAND_INTEL:
                    for sklearnex_algorithm in sklearnex_algorithms:
                        if (estimator_name == 'LinearRegression' and sklearnex_algorithm == 'linear') or estimator_name.lower() == sklearnex_algorithm:
                            logger.info('\nSklearn Algorithm :: {}, Sklearn Intel(R) Ex Algorithm :: {}'.format(estimator_name, sklearnex_algorithm))
                            if sklearn_is_patched(name=sklearnex_algorithm):
                                regressor = estimator_class()
                                estimator_execution_device = CPU_ACCELERATION_INTEL
                            break

                # Normal CPU processing if acceleration extensions are not applied
                if regressor is None:
                    regressor = estimator_class()

                # TODO: Revisit this code. sklearn throws error without n_components=1
                if estimator_name in ['CCA', 'PLSCanonical']:
                    regressor.set_params(n_components=1)

                # Hyperparameter tuning is requested
                if tune_all_models or estimator_name in models_to_tune:
                    estimator_params = regressor.get_params()

                    logger.info('Available hyperparameters to be tuned :: {}'.format(estimator_params))
                    logger.debug('Python type() for hyperparameters :: {}'.format(type(estimator_params)))

                    if estimator_execution_device == GPU_ACCELERATION_NVIDIA:
                        default_hyperparameters_module = bluemist.regression.tuning.cuml
                    else:
                        default_hyperparameters_module = bluemist.regression.tuning.sklearn

                    default_hyperparameters = getattr(default_hyperparameters_module, 'default_params', None)
                    model_params_for_tuning = getattr(default_hyperparameters_module, estimator_name, None)
                    remove_params = model_params_for_tuning.get('remove_params')

                    hyperparameters_to_remove = []
                    for hyperparameter, default_value in estimator_params.items():
                        if str(default_value) == 'deprecated':
                            hyperparameters_to_remove.append(hyperparameter)
                            logger.debug('Deprecated hyperparameter identified :: {}'.format(hyperparameter))
                        elif model_params_for_tuning is not None and hyperparameter in model_params_for_tuning:
                            estimator_params[hyperparameter] = model_params_for_tuning[hyperparameter]
                            logger.debug('Hyperparameter in model configuration :: {} :: {}'.format(hyperparameter, estimator_params[hyperparameter]))
                        elif hyperparameter in default_hyperparameters:
                            estimator_params[hyperparameter] = default_hyperparameters[hyperparameter]
                            logger.debug('Hyperparameter in default configuration :: {} :: {}'.format(hyperparameter, estimator_params[hyperparameter]))

                        if remove_params is not None:
                            if hyperparameter in remove_params and hyperparameter not in hyperparameters_to_remove:
                                hyperparameters_to_remove.append(hyperparameter)
                                logger.debug('Unsupported hyperparameter identified :: {}'.format(hyperparameter))

                    for hyperparameter_to_remove in hyperparameters_to_remove:
                        estimator_params.pop(hyperparameter_to_remove, None)

                    # Creating new dictionary of hyperparameters to add step name as required by the sklearn pipeline
                    hyperparameters = {}
                    for hyperparameter in estimator_params:
                        old_key = hyperparameter

                        if target_scaling_strategy is not None:
                            new_key = estimator_name + '__regressor__' + hyperparameter
                        else:
                            new_key = estimator_name + '__' + hyperparameter
                        hyperparameters[new_key] = estimator_params[old_key]

                    logger.info('Hyperparameters to be used for model tuning :: {}'.format(hyperparameters))

                    if target_scaling_strategy is not None:
                        # TODO: Remove the usage of TransformedTargetRegressor so speical handling is not required for cuML
                        transformed_target_regressor = TransformedTargetRegressor(regressor=regressor, transformer=target_scaler)

                        if environment.available_gpu == 'NVIDIA':
                            step_estimator = (estimator_name, regressor)
                        else:
                            step_estimator = (estimator_name, transformed_target_regressor)
                        steps = add_pipeline_step(estimator_name, step_estimator)
                    else:
                        step_estimator = (estimator_name, regressor)
                        steps = add_pipeline_step(estimator_name, step_estimator)

                    if steps is not None:
                        model_pipeline = Pipeline(steps=steps)
                        randomized_search = RandomizedSearchCV(model_pipeline, param_distributions=hyperparameters, n_iter=100, error_score='raise')
                        optimized_estimator = randomized_search.fit(X_train, y_train)
                        best_estimator_pipeline = optimized_estimator.best_estimator_

                        logger.debug('Model pipeline parameters :: {}'.format(model_pipeline.get_params().keys()))
                        logger.info('Fitted estimator with all parameters :: {}'.format(optimized_estimator))
                        logger.debug('Model pipeline :: {}'.format(model_pipeline))
                        logger.info('Model pipeline with best estimator :: {}'.format(best_estimator_pipeline))

                        if save_pipeline_to_disk:
                            logger.info('Saving model pipeline to disk')
                            save_model_pipeline(estimator_name, best_estimator_pipeline)
                else:
                    if target_scaling_strategy is not None:
                        # TODO: Remove the usage of TransformedTargetRegressor so speical handling is not required for cuML
                        transformed_target_regressor = TransformedTargetRegressor(regressor=regressor, transformer=target_scaler)

                        if environment.available_gpu == 'NVIDIA':
                            step_estimator = (estimator_name, regressor)
                        else:
                            step_estimator = (estimator_name, transformed_target_regressor)
                        steps = add_pipeline_step(estimator_name, step_estimator)
                    else:
                        step_estimator = (estimator_name, regressor)
                        steps = add_pipeline_step(estimator_name, step_estimator)

                    model_pipeline = Pipeline(steps=steps)
                    best_estimator_pipeline = model_pipeline.fit(X_train, y_train)

                    if save_pipeline_to_disk:
                        save_model_pipeline(estimator_name, best_estimator_pipeline)

                    logger.info('Model pipeline with best estimator (no hyperparameter tuning) :: {}'.format(best_estimator_pipeline))
                    logger.debug('Model pipeline (no hyperparameter tuning) :: {}'.format(model_pipeline))

                if tune_all_models or estimator_name in models_to_tune:
                    logger.info('Best score :: {}'.format(optimized_estimator.best_score_))
                    logger.info('Best hyperparameters :: {}'.format(optimized_estimator.best_params_))

                y_pred = best_estimator_pipeline.predict(X_test)

                # Convert cupy.ndarray to numpy.ndarray as sklearn returns numpy.ndarray but cuML will return cupy.ndarray
                y_pred_class_name = y_pred.__class__.__module__ + '.' + y_pred.__class__.__name__
                logger.info('y_pred_class_name :: {}'.format(y_pred_class_name))
                if y_pred_class_name == 'cupy.ndarray':
                    import cupy as cp
                    y_pred = cp.asnumpy(y_pred)

                scorer = metric_scorer(y_test, y_pred, metrics)
                estimator_stats_df = scorer.calculate_metrics()

                final_stats_df = estimator_stats_df.copy()
                execution_time = measure_execution_time(start_time)

                final_stats_df.insert(0, 'Estimator', estimator_name)  # Insert Estimator name as the first column in the dataframe
                final_stats_df.insert(1, 'Execution Device', estimator_execution_device)  # Insert Execution Device as the second column in the dataframe
                final_stats_df.insert(2, 'Execution Time', execution_time) # Insert Execution Time as the third column in the dataframe

                logger.info('Current estimator Stats :: \n{}'.format(final_stats_df.to_string()))
                logger.debug('Current estimator stats as dictionary :: {}'.format(final_stats_df.to_dict('records')))

                df = pd.concat([df, final_stats_df], ignore_index=True)
                logger.debug('Estimator stats so far :: \n{}'.format(df.to_string()))

                if experiment_name is not None:
                    mlflow = import_mlflow()
                    with mlflow.start_run(run_name=run_name):
                        logger.info('Capturing stats in MLFlow...')
                        mlflow.log_param('model', estimator_name)
                        mlflow.log_metrics(estimator_stats_df.to_dict('records')[0])
                        mlflow.sklearn.log_model(best_estimator_pipeline, 'model_' + estimator_name)
                        run_id = mlflow.active_run().info.run_id
                        logger.info('Model saved in run :: {}'.format(run_id))
            except Exception as e:
                exception = {'Estimator': [estimator_name], 'Exception': str(e)}
                exception_df = pd.DataFrame(exception)
                logger.info('Exception occurred :: \n{}'.format(exception_df))
                df = pd.concat([df, exception_df])
                logger.error('Exception occurred while training the model :: {}'.format(str(e)), exc_info=True)

    df.set_index('Estimator', inplace=True)
    #print(df)
    display(HTML(df.style
                 .highlight_max(
        subset=[col for col in df.columns if col.endswith('score') and is_numeric_dtype(df[col])], color='green')
                 .highlight_min(
        subset=[col for col in df.columns if col.endswith('score') and is_numeric_dtype(df[col])], color='yellow')
                 .highlight_max(
        subset=[col for col in df.columns if not col.endswith('score') and is_numeric_dtype(df[col])], color='yellow')
                 .highlight_min(
        subset=[col for col in df.columns if not col.endswith('score') and is_numeric_dtype(df[col])], color='green')
                 .to_html()))
    logger.info('Estimator stats across all trained models : \n{}'.format(df.to_string()))
