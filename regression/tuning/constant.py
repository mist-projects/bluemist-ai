import numpy as np

default_hyperparameters = {
    'alphas': [None],
    'alpha': [1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5],
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
    'copy_X_train': [True],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'cv': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'eps': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'fit_intercept': [True, False],
    'kernel': [None],
    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'l2_regularization': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'lambda_1': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'lambda_2': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'lambda_init': [None, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'learning_rate': np.arange(0.5, 50 + 1, 0.5).tolist(),
    'max_bins': np.arange(1, 255 + 1, 1).tolist(),
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'max_leaf_nodes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_samples': [0.7, 0.8, 0.9, 1],
    'max_features': [0.7, 0.8, 0.9, 1],
    'min_impurity_decrease': [0.0],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_weight_fraction_leaf': [0.0],
    'n_alphas': np.arange(100, 5000 + 1, 50).tolist(),
    'n_components': [1, 2, 3, 4, 5, 6],
    'n_estimators': np.arange(10, 1050 + 1, 50).tolist(),
    'n_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'n_iter_no_change': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'n_jobs': [-1],
    'normalize': [False],
    'normalize_y': [True, False],
    'oob_score': [True, False],
    'positive': [True, False],
    'precompute': [True, False],
    'quantile': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'random_state': [2],
    'scale': [True, False],
    'selection': ['cyclic', 'random'],
    'splitter': ['best', 'random'],
    'strategy': ['mean', 'median', 'quantile', 'constant'],
    'threshold_lambda': np.arange(5000.0, 20000.0 + 1, 1000.0).tolist(),
    'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'verbose': [False],
    'warm_start': [True, False]
}

AdaBoostRegressor = {
    'loss': ['linear', 'square', 'exponential']
}

DecisionTreeRegressor = {
    'max_features': [0.7, 0.8, 0.9, 1, 'auto', 'sqrt', 'log2']
}

ElasticNetCV = {
    'precompute': [None, True, False]
}

ExtraTreeRegressor = {
    'max_features': [0.7, 0.8, 0.9, 1, 'auto', 'sqrt', 'log2']
}

GaussianProcessRegressor = {
    'alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1],
    'optimizer': ['fmin_l_bfgs_b'],
    'n_restarts_optimizer': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

GradientBoostingRegressor = {
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'criterion': ['friedman_mse', 'squared_error', 'mse'],
    'init': [None, 'zero'],
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

HistGradientBoostingRegressor = {
    'categorical_features': [None],
    'early_stopping': ['auto', True, False],
    'loss': ['squared_error', 'absolute_error', 'poisson'],
    'monotonic_cst': [None],
    'scoring': [None, 'loss']
}
