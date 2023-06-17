# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.2
# Email: dew@bluemist-ai.one
# Created: May 29, 2023
# Last modified: June 16, 2023

import numpy as np

default_hyperparameters = {
    'output_type': ['array'],
    'random_state': [2],
    'verbose': [0],
    'dtype': [None],
    'n_jobs': [-1],
}

LinearRegression = {
    'algorithm': ['svd', 'eig', 'qr', 'svd-qr', 'svd-jacobi'],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'random_state': [2]
}

Ridge = {
    'alpha': [1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5],
    'solver': ['eig', 'svd', 'cd'],
    'fit_intercept': [True, False],
    'normalize': [True, False]
}

Lasso = {
    'alpha': [1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'solver': ['cd', 'qn'],
    'selection': ['cyclic', 'random']
}

ElasticNet = {
    'alpha': [1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5],
    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'solver': ['cd', 'qn'],
    'selection': ['cyclic', 'random']
}

MBSGDRegressor = {
    'alpha': [1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5],
    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'solver': ['cd', 'qn'],
    'selection': ['cyclic', 'random']
}

RandomForestRegressor = {
    'accuracy_metric': ['r2', 'media_ae', 'mean_ae', 'mse'],
    'bootstrap': [True, False],
    'class_weight': ['auto', 'balanced'],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'n_estimators': np.arange(10, 1050 + 1, 50).tolist(),
    'split_criterion': ['mse', 'poisson', 'gamma', 'inverse_gaussian'],
    'max_batch_size': [4096],
    'max_samples': [0.7, 0.8, 0.9, 1],
    'max_depth': [1, 2, 3, 4, 6, 8, 10, 12, 14, 16],
    'max_leaf_nodes': np.arange(2, 50 + 1, 1).tolist(),
    'max_leaves': np.arange(2, 50 + 1, 1).tolist(),
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_impurity_decrease': [0.0],
    'min_impurity_split': [1],
    'min_weight_fraction_leaf': [1],
    'n_bins': np.arange(2, 255 + 1, 1).tolist(),
    'n_streams': [4],
    'oob_score': [True, False],
    'warm_start': [True, False]
}

SVR = {
    'C': [1.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [1, 2, 3, 4, 5],
    'gamma': ['auto', 'scale'],
    'coef0': [0.0],
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'cache_size': [1024.0],
    'class_weight': [None, 'balanced'],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'multiclass_strategy': ['ovo', 'ovr'],
    'nochange_steps': [1000],
    'probability': [True, False]
}

KNeighborsRegressor = {
    'n_neighbors': [1, 2, 3, 4, 5, 6],
    'algorithm': ['auto', 'rbc', 'brute', 'ivfflat', 'ivfpq'],
    'metric': ['euclidean'],
    'weights': ['uniform']
}

KernelRidge = {
    'alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'],
    'gamma': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'coef0': [1],
    'degree': [1, 2, 3, 4, 5],
    'kernel_params': [None]
}
