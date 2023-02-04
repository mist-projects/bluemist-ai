"""
Main comment for regressor.py
"""

import importlib
import logging
import os
import traceback
from logging import config

import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import bluemist
from bluemist.pipeline.bluemist_pipeline import add_pipeline_step, save_pipeline
from bluemist.regression.constant import multi_output_regressors, multi_task_regressors, unsupported_regressors, \
    base_estimator_regressors

from bluemist.regression.tuning.constant import default_hyperparameters
from bluemist.utils.metrics import scoringStrategy
from sklearn.utils import all_estimators

from bluemist.utils.scaler import getScaler
from bluemist.utils import generate_api as generate_api
from bluemist.artifacts.api import predict


HOME_PATH = os.environ["HOME_PATH"]
config.fileConfig(HOME_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def get_regressor_class(module, class_name):
    module = importlib.import_module(module)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance


def initialize_mlflow(**kwargs):
    mlflow_experiment_name = kwargs.get('experiment_name')

    if mlflow_experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
        if not experiment:
            mlflow.create_experiment(name=mlflow_experiment_name,
                                     artifact_location='/home/shashank-agrawal/mlflow_artifact')

        if experiment is not None and experiment.lifecycle_stage == 'deleted':
            print('restore experiment')
            client = MlflowClient()
            client.restore_experiment(experiment.experiment_id)

        mlflow.set_experiment(mlflow_experiment_name)


def get_estimators(**kwargs):
    multi_output = kwargs.get('multi_output')
    multi_task = kwargs.get('multi_task')
    estimators = all_estimators(type_filter='regressor')
    print('estimators', estimators)

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

    print('estimators_to_remove >>', estimators_to_remove)

    for estimator_to_remove in estimators_to_remove:
        estimators.remove(estimator_to_remove)

    return estimators


def deploy_model(estimator_name, host, port):
    generate_api.generate_api_code(estimator_name=estimator_name)
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
    """

    tune_all_models = False
    tune_model_list = []
    target_scaler = None
    capture_stats = False

    if target_scaling_strategy is not None:
        target_scaler = getScaler(target_scaling_strategy)

    if isinstance(tune_models, str) and tune_models == 'all':
        tune_all_models = True
    elif isinstance(tune_models, list):
        tune_model_list = tune_models

    if experiment_name is not None:
        capture_stats = True
        initialize_mlflow(**locals())

    df = pd.DataFrame()

    estimators = get_estimators(**locals())

    i = 0
    for estimator_name, estimator_class in estimators:
        i = i + 1

        # if estimator_name == 'LinearRegression':
        if i == 20:
            try:
                print('Regressor Name', estimator_name)
                regressor = estimator_class()

                if tune_all_models or estimator_name in tune_model_list:
                    estimator_parameters = regressor.get_params()
                    print('parameters', type(estimator_parameters))
                    print('hyperparameter alpha_1', type(default_hyperparameters['alpha_1']))

                    model_hyperparameters_for_tuning = getattr(bluemist.regression.tuning.constant, estimator_name, None)

                    deprecated_keys = []
                    for key, value in estimator_parameters.items():
                        if value == 'deprecated':
                            deprecated_keys.append(key)
                            print('deprecated key', key)
                        elif model_hyperparameters_for_tuning is not None and key in model_hyperparameters_for_tuning:
                            estimator_parameters[key] = model_hyperparameters_for_tuning[key]
                            print('parameters[key]', estimator_parameters[key])
                        elif key in default_hyperparameters:
                            estimator_parameters[key] = default_hyperparameters[key]
                            print('parameters[key]', estimator_parameters[key])

                    for deprecated_key in deprecated_keys:
                        estimator_parameters.pop(deprecated_key, None)

                    # Creating new dictionary of hyperparameters to add step name as required by the pipeline
                    hyperparameters = {}

                    for key in estimator_parameters:
                        old_key = key

                        if target_scaling_strategy is not None:
                            new_key = estimator_name + '__regressor__' + key
                        else:
                            new_key = estimator_name + '__' + key
                        hyperparameters[new_key] = estimator_parameters[old_key]

                    print('Hyperparameters for Tuning :: ', hyperparameters)

                    if target_scaling_strategy is not None:
                        transformed_target_regressor = TransformedTargetRegressor(regressor=regressor,
                                                                                  transformer=target_scaler)
                        step_estimator = (estimator_name, transformed_target_regressor)
                        steps = add_pipeline_step(estimator_name, step_estimator)
                    else:
                        step_estimator = (estimator_name, regressor)
                        steps = add_pipeline_step(estimator_name, step_estimator)

                    if steps is not None:
                        pipe = Pipeline(steps=steps)
                        search = RandomizedSearchCV(pipe, param_distributions=hyperparameters)
                        tuned_estimator = search.fit(X_train, y_train)
                        pipeline_with_fitted_estimator = tuned_estimator.best_estimator_
                        print('pipeline keys ', pipe.get_params().keys())
                        print('fitted_estimator', tuned_estimator)
                        print('pipe', pipe)
                        print('pipeline_with_best_estimator', pipeline_with_fitted_estimator)
                        # print('pipeline step access', pipeline_with_fitted_estimator['BayesianRidge'])
                        if save_pipeline_to_disk:
                            save_pipeline(estimator_name, pipeline_with_fitted_estimator)

                        print('pipe1:', pipe)
                else:
                    if target_scaling_strategy is not None:
                        transformed_target_regressor = TransformedTargetRegressor(regressor=regressor,
                                                                                  transformer=target_scaler)
                        step_estimator = (estimator_name, transformed_target_regressor)
                        steps = add_pipeline_step(estimator_name, step_estimator)
                    else:
                        step_estimator = (estimator_name, regressor)
                        steps = add_pipeline_step(estimator_name, step_estimator)

                    pipe = Pipeline(steps=steps)
                    pipeline_with_fitted_estimator = pipe.fit(X_train, y_train)

                    if save_pipeline_to_disk:
                        save_pipeline(estimator_name, pipeline_with_fitted_estimator)

                    print('fitted_estimator', pipeline_with_fitted_estimator)
                    print('pipe2', pipe)

                if tune_all_models or estimator_name in tune_model_list:
                    print('Best Score: %s' % tuned_estimator.best_score_)
                    print('Best Hyperparameters: %s' % tuned_estimator.best_params_)

                y_pred = pipeline_with_fitted_estimator.predict(X_test)

                scorer = scoringStrategy(y_test, y_pred, metrics)
                estimator_stats_df = scorer.getStats()

                final_stats_df = estimator_stats_df.copy()

                # Insert Estimator name as the first column in the dataframe
                final_stats_df.insert(0, 'Estimator', estimator_name)
                print('stats_df', final_stats_df)

                df = pd.concat([df, final_stats_df], ignore_index=True)
                print('after concat', final_stats_df)
                print(estimator_stats_df.to_dict('records'))

                if capture_stats:
                    with mlflow.start_run(run_name=run_name):
                        print('Inside mlflow')
                        mlflow.log_param('model', estimator_name)
                        mlflow.log_metrics(estimator_stats_df.to_dict('records')[0])
                        mlflow.sklearn.log_model(tuned_estimator, "model")
                        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
            except Exception as e:
                exception = {'Estimator': [estimator_name], 'Exception': str(e)}
                exception_df = pd.DataFrame(exception)
                print('metrics', exception_df)
                df = pd.concat([df, exception_df])
                traceback.print_exc()

    df.style.set_table_styles([{'selector': '',
                                'props': [('border',
                                           '10px solid yellow')]}])

    logger.info('Estimator Stats : {}'.format(df.to_string()))
