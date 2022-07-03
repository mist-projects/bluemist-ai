from sklearn.linear_model import LinearRegression


class regressor(LinearRegression):

    def fit(self, X_train, y_train, sample_weight=None):
        print('Inside custom regressor')
        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)
        return regression_model

