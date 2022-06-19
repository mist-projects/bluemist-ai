regression_algorithms = []

# Linear regression
linear_regression = {'name': 'lr', 'provider': 'sklearn.linear_model.LinearRegression',
                     'module': 'regression.linear', 'class': 'regressor'}
regression_algorithms.append(linear_regression)

# Lasso regression
ridge_regression = {'name': 'lasso', 'provider': 'sklearn.linear_model.Lasso', 'module': 'regression.lasso',
                    'class': 'regressor'}
regression_algorithms.append(ridge_regression)

# Ridge regression
ridge_regression = {'name': 'ridge', 'provider': 'sklearn.linear_model.Ridge', 'module': 'regression.ridge',
                    'class': 'regressor'}
regression_algorithms.append(ridge_regression)

# RidgeCV regression
ridge_cv_regression = {'name': 'ridge_cv', 'provider': 'sklearn.linear_model.RidgeCV',
                       'module': 'regression.ridge_cv', 'class': 'regressor'}
regression_algorithms.append(ridge_cv_regression)

# SGDRegressor regression
sgd_regression = {'name': 'sgd', 'provider': 'sklearn.linear_model.SGDRegressor', 'module': 'regression.sgd',
                  'class': 'regressor'}
regression_algorithms.append(sgd_regression)
