import numpy as np

default_hyperparameters = {
    'alpha_1': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'alpha_2': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'base_estimator': [None],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
    'compute_score': [False],
    'copy_X': [True],
    'fit_intercept': [True, False],
    'lambda_1': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'lambda_2': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'learning_rate': np.arange(0.5, 50 + 1, 0.5).tolist(),
    'max_samples': [0.7, 0.8, 0.9, 1],
    'max_features': [0.7, 0.8, 0.9, 1],
    'n_estimators': np.arange(10, 1050 + 1, 50).tolist(),
    'n_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'normalize': [False],
    'oob_score': [True, False],
    'random_state': [2],
    'threshold_lambda': np.arange(5000.0, 20000.0 + 1, 1000.0).tolist(),
    'tol': [0.0001, 0.001, 0.01, 0.1],
    'verbose': [False],
    'warm_start': [True, False],
    'n_jobs': [-1]
}

AdaBoostRegressor = {
    'loss': ['linear', 'square', 'exponential']
}
