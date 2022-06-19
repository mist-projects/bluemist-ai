from sklearn.linear_model import RidgeCV

class regressor:

    def train_model(self, X_train, X_test, y_train, y_test):
        regression_model = RidgeCV()
        regression_model.fit(X_train, y_train)
        print(regression_model.score(X_test, y_test))
        return regression_model


