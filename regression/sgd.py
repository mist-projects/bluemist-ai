import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import scale


class regressor:

    def train_model(self, X_train, X_test, y_train, y_test):
        regression_model = SGDRegressor()

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)

        regression_model.fit(X_train, y_train)
        print(regression_model.score(X_test, y_test))
        return regression_model
