"""
Main comment for regressor.py
"""

import importlib
import logging
import traceback
from logging import config

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import regression.tuning
import utils.scaler
from pipeline.bluemist_pipeline import add_pipeline_step, save_pipeline
from regression.constant import multi_output_regressors, multi_task_regressors, unsupported_regressors, \
    base_estimator_regressors
from regression.overridden_estimators import overridden_regressors
from regression.tuning.constant import default_hyperparameters
from utils.metrics import scoringStrategy
from sklearn.utils import all_estimators

import mlflow
import mlflow.sklearn

config.fileConfig('logging.config')
logger = logging.getLogger("root")


def get_regressor_class(module, class_name):
    module = importlib.import_module(module)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance


def train_test_evaluate(
        data,
        tune_models=None,
        test_size=0.25,
        random_state=None,
        metrics='default',
        multi_output=False,
        multi_task=False,
        scale_data=False,
        scaling_type='StandardScaler',
        scale_target=False,
        save_pipeline_to_disk=True,
        mlflow_stats=False,
        mlflow_experiment_name=None,
        mlflow_run_name=None):
    """
    data: dataframe-like = None
        Dataframe to be passed to the ML estimator
    tune_models: bool, default=False
        Set to True to perform hyperparamter tuning
    test_size: float or int, default=0.25
        Proportion of the dataset to include in the test split.
    random_state: int, default=None
        random_state description
    metrics: dataframe-like = None
        Add description
    multi_output: bool, default=False
        Future use
    multi_task: bool, default=False
        Future use        
    scale_data: dataframe-like = None
        Add description
    scaling_type: dataframe-like = None
        Add description. Ignored if scale_data is False
    scale_target: dataframe-like = None
        Add description. Ignored if scale_data is False
    mlflow_stats: bool, default=False
        Set to True to log experiments in MLFlow
    mlflow_experiment_name: dataframe-like = None
        Add description. Ignored if mlflow_stats is False
    mlflow_run_name: dataframe-like = None
        Add description. Ignored if mlflow_stats is False
    """

    tune_all_models = False
    tune_model_list = []
    scaler = None

    # X = data.drop(['mpg', 'origin_europe'], axis=1)
    X = data.drop(['mpg'], axis=1)
    print('X columns', X.columns)
    # the dependent variable
    y = data[['mpg']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    if isinstance(tune_models, str):
        if tune_models == 'all':
            tune_all_models = True
    elif isinstance(tune_models, list):
        tune_model_list = tune_models

    df = pd.DataFrame()

    if scale_data:
        scaler = utils.scaler.getScaler(scaling_type)
    if scaler is None:
        scale_target = False

    if mlflow_stats:
        if mlflow_experiment_name is not None:
            experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
            if not experiment:
                mlflow.create_experiment(name=mlflow_experiment_name,
                                         artifact_location='/home/shashank-agrawal/mlflow_artifact')

            if experiment.lifecycle_stage == 'deleted':
                client = MlflowClient()
                client.restore_experiment(experiment.experiment_id)

            mlflow.set_experiment(mlflow_experiment_name)

    estimators = all_estimators(type_filter='regressor')
    print('estimators', estimators)

    estimators_to_remove = []
    for estimator in estimators:
        if not multi_output:
            if estimator[0] in multi_output_regressors:
                estimators_to_remove.append(estimator)
        if not multi_task:
            if estimator[0] in multi_task_regressors:
                estimators_to_remove.append(estimator)
        if unsupported_regressors:
            if estimator[0] in unsupported_regressors:
                estimators_to_remove.append(estimator)
        if base_estimator_regressors:
            if estimator[0] in base_estimator_regressors:
                estimators_to_remove.append(estimator)

    print('estimators_to_remove >>', estimators_to_remove)

    for estimator_to_remove in estimators_to_remove:
        estimators.remove(estimator_to_remove)

    print('estimators after removal >>', estimators)
    custom_regressors_df = overridden_regressors

    i = 0
    for estimator_name, RegressorClass in estimators:
        i = i + 1

        # if estimator_name == 'LinearRegression':
        if i == 14:
            try:
                print('Regressor Name', estimator_name)

                custom_estimator = custom_regressors_df.query("Name == @estimator_name")
                if not custom_estimator.empty:
                    reg = get_regressor_class(custom_estimator['Module'].values[0], custom_estimator['Class'].values[0])
                    print('Custom regressor', reg)
                else:
                    reg = RegressorClass()

                if tune_all_models or estimator_name in tune_model_list:
                    estimator_parameters = reg.get_params()
                    print('parameters', type(estimator_parameters))
                    print('hyperparameter alpha_1', type(default_hyperparameters['alpha_1']))
                    # default_hyperparameters_for_tuning = default_hyperparameters
                    model_hyperparameters_for_tuning = getattr(regression.tuning.constant, estimator_name, None)

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

                        if scale_target:
                            new_key = estimator_name + '__regressor__' + key
                        else:
                            new_key = estimator_name + '__' + key
                        hyperparameters[new_key] = estimator_parameters[old_key]

                    print('Hyperparameters for Tuning :: ', hyperparameters)

                    if scaler is not None:
                        step_scale = ('scaler', scaler)
                        add_pipeline_step(estimator_name, step_scale)

                    if scale_target:
                        tt = TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
                        step_estimator = (estimator_name, tt)
                        steps = add_pipeline_step(estimator_name, step_estimator)
                    else:
                        step_estimator = (estimator_name, reg)
                        steps = add_pipeline_step(estimator_name, step_estimator)

                    if steps is not None:
                        pipe = Pipeline(steps=steps)
                        search = RandomizedSearchCV(pipe, param_distributions=hyperparameters)
                        fitted_estimator = search.fit(X_train, y_train)
                        pipeline_with_best_estimator = fitted_estimator.best_estimator_
                        # fitted_estimator = search.best_estimator_
                        print('fitted_estimator', fitted_estimator)
                        print('pipe', pipe)
                        print('pipeline_with_best_estimator', pipeline_with_best_estimator)

                        if save_pipeline_to_disk:
                            save_pipeline(estimator_name, pipeline_with_best_estimator)

                        print('pipe1:', pipe)
                else:
                    if scaler is not None:
                        step_scale = ('scaler', scaler)
                        add_pipeline_step(estimator_name, step_scale)

                    if scale_target:
                        tt = TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
                        step_estimator = (estimator_name, tt)
                        steps = add_pipeline_step(estimator_name, step_estimator)
                    else:
                        step_estimator = (estimator_name, reg)
                        steps = add_pipeline_step(estimator_name, step_estimator)

                    pipe = Pipeline(steps=steps)
                    fitted_estimator = pipe.fit(X_train, y_train)

                    if save_pipeline_to_disk:
                        save_pipeline(estimator_name, fitted_estimator)

                    print('fitted_estimator', fitted_estimator)
                    print('pipe2', pipe)

                if tune_all_models or estimator_name in tune_model_list:
                    print('Best Score: %s' % fitted_estimator.best_score_)
                    print('Best Hyperparameters: %s' % fitted_estimator.best_params_)

                    y_pred = fitted_estimator.predict(X_test)
                else:
                    y_pred = fitted_estimator.predict(X_test)

                scorer = scoringStrategy(y_test, y_pred, metrics)
                estimator_stats_df = scorer.getStats()

                final_stats_df = estimator_stats_df.copy()

                # Insert Estimator name as the first column in the dataframe
                final_stats_df.insert(0, 'Estimator', estimator_name)
                print('stats_df', final_stats_df)

                df = pd.concat([df, final_stats_df], ignore_index=True)
                print('after concat', final_stats_df)
                print(estimator_stats_df.to_dict('records'))

                if mlflow_stats:
                    with mlflow.start_run(run_name=mlflow_run_name):
                        print('Inside mlflow')
                        mlflow.log_param('model', estimator_name)
                        mlflow.log_metrics(estimator_stats_df.to_dict('records')[0])
                        mlflow.sklearn.log_model(fitted_estimator, "model")
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
