import pandas as pd
regression_algorithms = []

# Linear regression
linear_regression = ['LinearRegression', 'regression.linear', 'regressor']
regression_algorithms.append(linear_regression)

overridden_regressors = pd.DataFrame(regression_algorithms, columns=['Name', 'Module', 'Class'])

