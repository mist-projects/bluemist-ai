
__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


import numpy as np

default_hyperparameters = {
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'alphas': [None],
    'alpha': [1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5],
    'alpha_1': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'alpha_2': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'alpha_init': [None, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],  # TODO: Review this parameter
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
    'ccp_alpha': [0.0],
    'compute_score': [False],
    'constant': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'copy': [True],
    'copy_X': [True],
    'copy_X_train': [True],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'cv': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'degree': [1, 2, 3, 4, 5],
    'dual': [True, False],
    'early_stopping': [True, False],
    'eps': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'estimator': [None],
    'fit_intercept': [True, False],
    'fit_path': [True, False],
    'jitter': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'kernel': [None],
    'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'l2_regularization': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'lambda_1': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'lambda_2': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'lambda_init': [None, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],  # TODO: Review this parameter
    'leaf_size': np.arange(1, 40 + 1, 1).tolist(),
    'learning_rate': np.arange(0.1, 50 + 1, 0.5).tolist(),
    'max_bins': np.arange(2, 255 + 1, 1).tolist(),
    'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_features': [0.7, 0.8, 0.9, 1],
    'max_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'max_leaf_nodes': np.arange(2, 50 + 1, 1).tolist(),
    'max_n_alphas': np.arange(100, 5000 + 1, 100).tolist(),
    'max_samples': [0.7, 0.8, 0.9, 1],
    'max_subpopulation': [1e6, 1e5, 10000, 1000, 100, 1],
    'metric': ['minkowski'],
    'metric_params': [None],
    'min_impurity_decrease': [0.0],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_weight_fraction_leaf': [0.0],
    'n_alphas': np.arange(100, 5000 + 1, 50).tolist(),
    'n_components': [1],
    'n_estimators': np.arange(10, 1050 + 1, 50).tolist(),
    'n_iter': np.arange(100, 5000 + 1, 50).tolist(),
    'n_iter_no_change': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'n_jobs': [-1],
    'n_neighbors': np.arange(1, 10 + 1, 1).tolist(),
    'n_subsamples': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'normalize': [False],
    'normalize_y': [True, False],
    'oob_score': [True, False],
    'p': [1, 2, 3, 4, 5],
    'positive': [True, False],
    'precompute': [True, False],
    'quantile': [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'random_state': [2],
    'scale': [True, False],
    'selection': ['cyclic', 'random'],
    'shuffle': [True, False],
    'solver': ['lbfgs', 'newton-cholesky'],
    'splitter': ['best', 'random'],
    'strategy': ['mean', 'median', 'quantile', 'constant'],
    'threshold_lambda': np.arange(5000.0, 20000.0 + 1, 1000.0).tolist(),
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'verbose': [False],
    'warm_start': [True, False],
    'weights': [None, 'uniform', 'distance']
}

AdaBoostRegressor = {
    'loss': ['linear', 'square', 'exponential']
}

DecisionTreeRegressor = {
    'max_features': [0.7, 0.8, 0.9, 1, 'sqrt', 'log2']
}

ElasticNetCV = {
    'precompute': ['auto', True, False],
    'cv': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

ExtraTreeRegressor = {
    'max_features': [None, 0.7, 0.8, 0.9, 1, 'sqrt', 'log2']
}

GaussianProcessRegressor = {
    'alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1],
    'optimizer': [None, 'fmin_l_bfgs_b'],
    'n_restarts_optimizer': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

GradientBoostingRegressor = {
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'criterion': ['friedman_mse', 'squared_error'],
    'init': [None, 'zero'],
    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
    'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

HistGradientBoostingRegressor = {
    'categorical_features': [None],
    'early_stopping': ['auto', True, False],
    'loss': ['squared_error', 'absolute_error', 'poisson', 'quantile'],
    'monotonic_cst': [None],
    'scoring': [None, 'loss'],
    'interaction_cst': ['pairwise', 'no_interactions'],
    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'min_samples_leaf': np.arange(1, 20 + 1, 1).tolist()
}

HuberRegressor = {
    'epsilon': [1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    'alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1]
}

KernelRidge = {
    'alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1],
    'kernel': ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'],
    'gamma': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'coef0': [1],
    'kernel_params': [None]
}

Lars = {
    'normalize': [True, False],
    'precompute': ['auto', True, False],
    'n_nonzero_coefs': np.arange(100, 1000 + 1, 100).tolist()
}

LarsCV = {
    'normalize': [True, False],
    'precompute': ['auto', True, False],
    'cv': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

LassoCV = {
    'cv': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

LassoLarsCV = {
    'cv': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'eps': [2.220446049250313e-16, 1e-10, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1]
}

LassoLarsIC = {
    'criterion': ['aic', 'bic'],
    'noise_variance': [None]
}

LinearSVR = {
    'epsilon': [0.0],
    'C': [1.0],
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    'intercept_scaling': [1.0]
}

MLPRegressor = {
    'hidden_layer_sizes': [(100,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'batch_size': ['auto'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'learning_rate_init': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'power_t': [0.5],
    'momentum': [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'nesterovs_momentum': [True, False],
    'validation_fraction': [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'beta_1': [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'beta_2': [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999],
    'epsilon': [1e-8, 1e-7, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'max_fun': [1000, 3000, 6000, 9000, 12000, 15000]
}

NuSVR = {
    'nu': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'C': [1.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'degree': [1, 2, 3, 4, 5],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0],
    'shrinking': [True, False],
    'tol': [1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'cache_size': [200],
    'max_iter': [1, 2, 3, 4, 5]
}

OrthogonalMatchingPursuit = {
    'n_nonzero_coefs': [None],
    'tol': [None]
}

OrthogonalMatchingPursuitCV = {
    'cv': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'max_iter': [5]
}

PLSCanonical = {
    'algorithm': ['nipals', 'svd']
}

PassiveAggressiveRegressor = {
    'C': [1.0],
    'n_iter_no_change': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    'epsilon': [1e-8, 1e-7, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'average': [True, False]
}

QuantileRegressor = {
    'solver': ['highs-ds', 'highs-ipm', 'highs', 'interior-point', 'revised simplex'],
    'solver_options': [None],
    'quantile': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}

RANSACRegressor = {
    'min_samples': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'residual_threshold': [None],
    'is_data_valid': [None],
    'is_model_valid': [None],
    'max_trials': np.arange(100, 1000 + 1, 50).tolist(),
    'max_skips': [np.inf],
    'stop_n_inliers': [np.inf],
    'stop_score': [np.inf],
    'stop_probability': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0],
    'loss': ['absolute_error', 'squared_error']
}

RadiusNeighborsRegressor = {
    'radius': [1.0],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': np.arange(2, 50 + 1, 1).tolist(),
    'p': [1, 2],
    'metric': ['euclidean', 'minkowski'],
    'metric_params': [None]
}

RandomForestRegressor = {
    'max_features': [None, 'sqrt', 'log2', 0.7, 0.8, 0.9, 1]
}

Ridge = {
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
}

RidgeCV = {
    'alphas': [(0.1, 1.0, 10.0)],
    'scoring': [None],
    'gcv_mode': ['auto', 'svd', 'eigen'],
    'store_cv_values': [False],
    'alpha_per_target': [False],
    'cv': [2, 3, 4, 5, 6, 7, 8, 9, 10],
}

SGDRegressor = {
    'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet', None],
    'alpha': [1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'l1_ratio': [0.15],
    'epsilon': [1e-8, 1e-7, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.01],
    'power_t': [0.25],
    'validation_fraction': [0.1],
    'average': [True, False]
}

SVR = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'coef0': [0.0],
    'C': [1.0],
    'epsilon': [1e-8, 1e-7, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1],
    'shrinking': [True, False],
    'cache_size': [200]
}

TransformedTargetRegressor = {
    'regressor': [None],
    'transformer': [None],
    'func': [None],
    'inverse_func': [None],
    'check_inverse': [True, False]
}

TweedieRegressor = {
    'power': [0, 1, 2, 3],
    'alpha': [0.0, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5],
    'link': ['auto', 'identity', 'log']
}
