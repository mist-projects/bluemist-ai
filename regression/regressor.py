import importlib
import traceback

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn import pipeline, set_config
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

import regression.tuning
from pipeline.bluemist_pipeline import add_pipeline_step, save_pipeline
from regression.overridden_estimators import overridden_regressors
from regression.tuning.constant import default_hyperparameters
from utils.metrics import scoringStrategy
from sklearn.utils import all_estimators

import mlflow
import mlflow.sklearn


def get_regressor_class(module, class_name):
    module = importlib.import_module(module)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance


def train_test_evaluate(data, tune_models=False, test_size=0.25, random_state=2, metrics='default',
                        mlflow_experiment_name=None, mlflow_run_name=None):
    X = data.drop(['mpg', 'origin_europe'], axis=1)
    # the dependent variable
    y = data[['mpg']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    tune_all_models = False
    tune_model_list = []

    if isinstance(tune_models, str):
        if tune_models == 'all':
            tune_all_models = True
    elif isinstance(tune_models, list):
        tune_model_list = tune_models

    df = pd.DataFrame()

    estimators = all_estimators(type_filter='regressor')
    custom_regressors_df = overridden_regressors

    if mlflow_experiment_name is not None:
        experiment = mlflow.get_experiment_by_name('TEST2')
        if not experiment:
            mlflow.create_experiment(name='TEST2', artifact_location='/home/shashank-agrawal/mlflow_artifact')

        if experiment.lifecycle_stage == 'deleted':
            client = MlflowClient()
            client.restore_experiment(experiment.experiment_id)

        mlflow.set_experiment('TEST2')

    i = 0
    for name, RegressorClass in estimators:
        i = i + 1

        if i <= 2:
            try:
                print('Regressor Name', name)

                custom_estimator = custom_regressors_df.query("Name == @name")
                if not custom_estimator.empty:
                    reg = get_regressor_class(custom_estimator['Module'].values[0], custom_estimator['Class'].values[0])
                else:
                    reg = RegressorClass()

                y_train = np.ravel(y_train)
                y_test = np.ravel(y_test)

                if tune_all_models or name in tune_model_list:
                    estimator_parameters = reg.get_params()
                    print('parameters', type(estimator_parameters))
                    print('hyperparameter alpha_1', type(default_hyperparameters['alpha_1']))
                    default_hyperparameters_for_tuning = default_hyperparameters
                    model_hyperparameters_for_tuning = getattr(regression.tuning.constant, name, None)

                    deprecated_keys = []
                    for key, value in estimator_parameters.items():
                        if value == 'deprecated':
                            deprecated_keys.append(key)
                            print('deprecated key', key)
                        elif model_hyperparameters_for_tuning is not None and key in model_hyperparameters_for_tuning:
                            estimator_parameters[key] = model_hyperparameters_for_tuning[key]
                            print('parameters[key]', estimator_parameters[key])
                        elif key in default_hyperparameters_for_tuning:
                            estimator_parameters[key] = default_hyperparameters_for_tuning[key]
                            print('parameters[key]', estimator_parameters[key])

                    for deprecated_key in deprecated_keys:
                        estimator_parameters.pop(deprecated_key, None)

                    # Creating new dictionary of hyperparameters to add step name as required by the pipeline
                    hyperparameters = {}

                    for key in estimator_parameters:
                        old_key = key
                        new_key = name + '__' + key
                        hyperparameters[new_key] = estimator_parameters[old_key]

                    print('Hyperparameters for Tuning :: ', hyperparameters)

                    step_estimator = (name, reg)
                    steps = add_pipeline_step(name, step_estimator)
                    # steps = [step_estimator]
                    pipe = Pipeline(steps=steps)
                    search = RandomizedSearchCV(pipe, param_distributions=hyperparameters)
                    fitted_estimator = search.best_estimator_.fit(X_train, y_train)
                    save_pipeline(name, pipe)
                    print(pipe)
                else:
                    step_estimator = (name, reg)
                    steps = [step_estimator]
                    pipe = Pipeline(steps=steps)
                    print(pipe)
                    fitted_estimator = pipe.fit(X_train, y_train)

                if tune_all_models or name in tune_model_list:
                    print('Best Score: %s' % fitted_estimator.best_score_)
                    print('Best Hyperparameters: %s' % fitted_estimator.best_params_)

                    y_pred = fitted_estimator.predict(X_test)
                else:
                    y_pred = fitted_estimator.predict(X_test)

                scorer = scoringStrategy(y_test, y_pred, metrics)
                stats_df = scorer.getStats()

                final_stats_df = stats_df.copy()

                # Insert Estimator name as the first column in the dataframe
                final_stats_df.insert(0, 'Estimator', name)
                print('stats_df', final_stats_df)

                df = pd.concat([df, final_stats_df], ignore_index=True)
                print('after concat', final_stats_df)
                print(stats_df.to_dict('records'))
                with mlflow.start_run(run_name=mlflow_run_name):
                    print('Inside mlflow')
                    mlflow.log_param('model', name)
                    mlflow.log_metrics(stats_df.to_dict('records')[0])
                    mlflow.sklearn.log_model(fitted_estimator, "model")
                    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
            except Exception as e:
                metrics = {'Estimator': [name], 'mae': None, 'mse': None, 'rmse': None, 'r2_score': None}
                df_metrics = pd.DataFrame(metrics)
                print('metrics', df_metrics)
                df = pd.concat([df, df_metrics])
                traceback.print_exc()

    print(df)
