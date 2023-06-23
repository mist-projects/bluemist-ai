# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.2
# Email: dew@bluemist-ai.one
# Created: May 29, 2023
# Last modified: June 18, 2023

import numpy as np

default_params = {
    'dtype': [None],
    'output_type': ['array'],
    'random_state': [2],
    'verbose': [4]
}

ElasticNet = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'selection': ['cyclic', 'random'],
    'remove_params': ['handle', 'solver']
}

KernelRidge = {
    'kernel': ['additive_chi2', 'cosine', 'laplacian', 'linear', 'poly', 'polynomial', 'rbf', 'sigmoid'],  # 'chi2' is not included as its throwing error
    'gamma': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'coef0': [1],
    'degree': [3],
    'remove_params': ['handle', 'alpha', 'kernel_params']
}

KNeighborsRegressor = {
    'algorithm': ['auto'],
    'algo_params': [None],
    'metric': ['euclidean'],
    'metric_params': [None],
    'n_jobs': [None],
    'n_neighbors': [1, 2, 3, 4, 5, 6],
    'p': [1, 2, 3, 4, 5],
    'weights': ['uniform'],
    'remove_params': ['handle']
}

LinearRegression = {
    'algorithm': ['svd', 'eig', 'qr', 'svd-qr', 'svd-jacobi'],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'remove_params': ['handle']
}

LinearSVR = {
    'penalty': ['l1', 'l2'],
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    'fit_intercept': [True, False],
    'penalized_intercept': [True, False],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'linesearch_max_iter': np.arange(100, 1000 + 1, 50).tolist(),
    'lbfgs_memory': [5, 10, 20, 30],
    'C': [1.0],
    'grad_tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'change_tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'epsilon': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 0],  # TODO: Revisit this variable in other estimators
    'remove_params': ['handle']
}

Lasso = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # TODO: Revisit this variable in other estimators
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'selection': ['cyclic', 'random'],
    'remove_params': ['handle', 'solver']  # solver is throwing error hence removing during hyperparameter tuning
}

MBSGDRegressor = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'solver': ['cd', 'qn'],
    'selection': ['cyclic', 'random']
}

RandomForestRegressor = {
    'accuracy_metric': ['r2'],
    'bootstrap': [True],
    'class_weight': [None],
    'criterion': [None],
    'split_criterion': [2], # 0 or 'gini' for gini impurity; 1 or 'entropy' for information gain (entropy); 2 or 'mse' for mean squared error; 4 or 'poisson' for poisson half deviance; 5 or 'gamma' for gamma half deviance; 6 or 'inverse_gaussian' for inverse gaussian deviance
    'max_batch_size': [4096],
    'max_depth': [2, 4, 8, 12, 16],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_leaf_nodes': [None],
    'max_leaves': [-1],
    'max_samples': [0.7, 0.8, 0.9, 1.0],
    'min_samples_leaf': np.arange(1, 10, 1).tolist(),
    'min_samples_split': np.arange(2, 21, 3).tolist(),
    'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'min_impurity_split': [None],
    'min_weight_fraction_leaf': [None],
    'n_bins': [128],
    'n_estimators': [100],
    'n_jobs': [None],
    'n_streams': [4],
    'oob_score': [None],
    'warm_start': [None],
    'remove_params': ['handle']
}

Ridge = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'solver': ['eig', 'svd'],  # 'cd' is not included as its throwing error
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'remove_params': ['handle']
}

SVR = {
    'C': [1.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [1, 2, 3, 4, 5],
    'gamma': ['auto', 'scale'],
    'coef0': [0.0],
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'epsilon': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 0],  # TODO: Revisit this variable in other estimators
    'cache_size': [1024.0],
    'class_weight': [None, 'balanced'],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'multiclass_strategy': ['ovo', 'ovr'],
    'nochange_steps': [1000],
    'probability': [True, False],
    'remove_params': ['handle']
}
