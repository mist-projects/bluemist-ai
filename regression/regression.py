import importlib

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

import regression.tuning
from regression.overridden_estimators import overridden_regressors
from utils.metrics import scoingStrategy
from sklearn.utils import all_estimators


def getRegressorClass(module, class_name):
    module = importlib.import_module(module)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance


class regressor:

    def train_model(X_train, X_test, y_train, y_test):
        df = pd.DataFrame()

        estimators = all_estimators(type_filter='regressor')
        custom_regressors_df = overridden_regressors;

        for name, RegressorClass in estimators:
            if name == 'ARDRegression':
                try:
                    print('Appending name', name)
                    print('Appending class', RegressorClass)

                    estimator = custom_regressors_df.query("Name == @name")
                    if not estimator.empty:
                        reg = getRegressorClass(estimator['Module'].values[0], estimator['Class'].values[0])
                    else:
                        reg = RegressorClass()

                    parameters = reg.get_params()
                    print('parameters', parameters)
                    print('hyperparameter alpha_1', regression.tuning.hyperparameters['alpha_1'])

                    print('parameters', type(parameters))
                    print('hyperparameter alpha_1', type(regression.tuning.hyperparameters['alpha_1']))
                    available_hyperparameters_for_tuning = regression.tuning.hyperparameters

                    deprecated_keys = []
                    for key, value in parameters.items():
                        if value == 'deprecated':
                            deprecated_keys.append(key)
                            print('deprecated key', key)
                        if key in available_hyperparameters_for_tuning:
                            parameters[key] = available_hyperparameters_for_tuning[key]
                            print('parameters[key]', parameters[key])

                    for deprecated_key in deprecated_keys:
                        parameters.pop(deprecated_key, None)

                    print('modified parameters', parameters)

                    y_train = np.ravel(y_train)
                    y_test = np.ravel(y_test)

                    search = RandomizedSearchCV(reg, parameters);
                    #fitted_estimator = reg.fit(X_train, y_train)
                    fitted_estimator = search.fit(X_train, y_train)
                    y_pred = fitted_estimator.predict(X_test);

                    print('Best Score: %s' % fitted_estimator.best_score_)
                    print('Best Hyperparameters: %s' % fitted_estimator.best_params_)

                    score = scoingStrategy(y_test, y_pred)

                    mae = score.get_mae()
                    r2_score = score.get_r2_score()
                    mse = score.get_mse()
                    rmse = score.get_rmse()

                    metrics = {'Estimator': [name], 'mae': [mae], 'mse': [mse], 'rmse': [rmse], 'r2_score': [r2_score], 'regression_error': None}
                    df_metrics = pd.DataFrame(metrics)
                    print('metrics', df_metrics)
                    df = pd.concat([df, df_metrics])
                except Exception as e:
                    metrics = {'Estimator': [name], 'mae': None, 'mse': None, 'rmse': None, 'r2_score': None, 'regression_error': [str(e)]}
                    df_metrics = pd.DataFrame(metrics)
                    print('metrics', df_metrics)
                    df = pd.concat([df, df_metrics])
                    print(e)

        print(df)
