import numpy as np

default_hyperparameters = {
    'alpha_1': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'alpha_2': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'alpha_init': [None, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'base_estimator': [None],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
    'ccp_alpha': [0.0],
    'compute_score': [False],
    'constant': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'copy': [True],
    'copy_X': [True],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'fit_intercept': [True, False],
    'lambda_1': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'lambda_2': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'lambda_init': [None, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'learning_rate': np.arange(0.5, 50 + 1, 0.5).tolist(),
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'max_leaf_nodes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_samples': [0.7, 0.8, 0.9, 1],
    'max_features': [0.7, 0.8, 0.9, 1],
    'min_impurity_decrease': [0.0],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_weight_fraction_leaf': [0.0],
    'n_components': [1, 2, 3, 4, 5, 6],
    'n_estimators': np.arange(10, 1050 + 1, 50).tolist(),
    'n_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'n_jobs': [-1],
    'normalize': [False],
    'oob_score': [True, False],
    'quantile': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'random_state': [2],
    'scale': [True, False],
    'splitter': ['best', 'random'],
    'strategy': ['mean', 'median', 'quantile', 'constant'],
    'threshold_lambda': np.arange(5000.0, 20000.0 + 1, 1000.0).tolist(),
    'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'verbose': [False],
    'warm_start': [True, False]
}

AdaBoostRegressor = {
    'loss': ['linear', 'square', 'exponential']
}

DecisionTreeRegressor = {
    'max_features': [0.7, 0.8, 0.9, 1, 'auto', 'sqrt', 'log2'],
}

