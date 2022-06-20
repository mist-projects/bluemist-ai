import numpy as np

hyperparameters = {
    'alpha_1': [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01],
    'alpha_2': [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01],
    'compute_score': [False],
    'copy_X': [True],
    'fit_intercept': [True, False],
    'lambda_1': [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01],
    'lambda_2': [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01],
    'n_iter': np.arange (100, 5000 + 1, 50).tolist(),
    'normalize': [False],
    'threshold_lambda': np.arange(5000.0, 20000.0 + 1, 1000.0).tolist(),
    'tol': [0.0001, 0.001, 0.01, 0.1],
    'verbose': [False]
}
