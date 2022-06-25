import importlib

import pandas as pd

from regression.overridden_estimators import regression_algorithms
from utils.metrics import scoringStrategy


class regressor:

    def train_model(X_train, X_test, y_train, y_test):
        all_algorithms = regression_algorithms
        df = pd.DataFrame()

        for algorithm in all_algorithms:
            print(algorithm['provider'])
            class_name = algorithm['class']
            module = algorithm['module']

            module = importlib.import_module(module)
            class_ = getattr(module, class_name)
            instance = class_()
            fitted_estimator = instance.train_model(X_train, X_test, y_train, y_test)
            y_pred = fitted_estimator.predict(X_test);

            # y_test = np.ravel(y_test.to_numpy())
            # y_pred = np.ravel(y_pred)

            print(y_test)
            print(y_pred)

            print(type(y_test))
            print(type(y_pred))

            score = scoringStrategy(y_test, y_pred)
            mae = score.get_mae()
            r2_score = score.get_r2_score()
            mse = score.get_mse()
            rmse = score.get_rmse()

            print('mae', mae)
            print('mse', mse)
            print('rmse', rmse)
            print('r2_score', r2_score)

            df_meta_data = pd.DataFrame([algorithm])
            metrics = {'mae': [mae], 'mse': [mse], 'rmse': [rmse], 'r2_score': [r2_score]}
            df_metrics = pd.DataFrame(metrics)
            print('metrics', df_metrics)

            df_temp = pd.concat([df_meta_data, df_metrics], axis=1)
            df = pd.concat([df, df_temp])

        print(df)
