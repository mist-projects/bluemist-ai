import importlib

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import regression.tuning
from regression.overridden_estimators import overridden_regressors
from utils.metrics import scoingStrategy
from sklearn.utils import all_estimators


def get_regressor_class(module, class_name):
    module = importlib.import_module(module)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance


def train_test_evaluate(data, tune_models, test_size=0.25, random_state=2):

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

    i = 0
    for name, RegressorClass in estimators:
        i = i + 1

        if i == 1:
            try:
                print('Appending name', name)
                print('Appending class', RegressorClass)

                estimator = custom_regressors_df.query("Name == @name")
                if not estimator.empty:
                    reg = get_regressor_class(estimator['Module'].values[0], estimator['Class'].values[0])
                else:
                    reg = RegressorClass()

                y_train = np.ravel(y_train)
                y_test = np.ravel(y_test)

                if tune_all_models or name in tune_model_list:
                    parameters = reg.get_params()
                    print('parameters', parameters)
                    print('hyperparameter alpha_1', regression.tuning.hyperparameters['alpha_1'])

                    print('parameters', type(parameters))
                    print('hyperparameter alpha_1', type(regression.tuning.hyperparameters['alpha_1']))
                    default_hyperparameters_for_tuning = regression.tuning.hyperparameters
                    model_hyperparameters_for_tuning = getattr(regression.tuning, name, None)

                    deprecated_keys = []
                    for key, value in parameters.items():
                        if value == 'deprecated':
                            deprecated_keys.append(key)
                            print('deprecated key', key)
                        elif model_hyperparameters_for_tuning is not None and key in model_hyperparameters_for_tuning:
                            parameters[key] = model_hyperparameters_for_tuning[key]
                            print('parameters[key]', parameters[key])
                        elif key in default_hyperparameters_for_tuning:
                            parameters[key] = default_hyperparameters_for_tuning[key]
                            print('parameters[key]', parameters[key])

                    for deprecated_key in deprecated_keys:
                        parameters.pop(deprecated_key, None)

                    print('modified parameters', parameters)

                    search = RandomizedSearchCV(reg, parameters)
                    fitted_estimator = search.fit(X_train, y_train)
                else:
                    fitted_estimator = reg.fit(X_train, y_train)

                y_pred = fitted_estimator.predict(X_test);

                if tune_all_models or name in tune_model_list:
                    print('Best Score: %s' % fitted_estimator.best_score_)
                    print('Best Hyperparameters: %s' % fitted_estimator.best_params_)

                score = scoingStrategy(y_test, y_pred)

                mae = score.get_mae()
                r2_score = score.get_r2_score()
                mse = score.get_mse()
                rmse = score.get_rmse()

                metrics = {'Estimator': [name], 'mae': [mae], 'mse': [mse], 'rmse': [rmse], 'r2_score': [r2_score],
                           'regression_error': None}
                df_metrics = pd.DataFrame(metrics)
                print('metrics', df_metrics)
                df = pd.concat([df, df_metrics])
            except Exception as e:
                metrics = {'Estimator': [name], 'mae': None, 'mse': None, 'rmse': None, 'r2_score': None,
                           'regression_error': [str(e)]}
                df_metrics = pd.DataFrame(metrics)
                print('metrics', df_metrics)
                df = pd.concat([df, df_metrics])
                print(e)

    print(df)
