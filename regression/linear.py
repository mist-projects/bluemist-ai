from sklearn.linear_model import LinearRegression


class regressor:

    def fit(self, X_train, y_train):
        print('Inside custom regressor')
        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)
        return regression_model
