# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.2
# Email: dew@bluemist-ai.one
# Created: May 29, 2023
# Last modified: May 29, 2023

default_hyperparameters = {
    'output_type': ['input']
}

LinearRegression = {
    'algorithm': ['svd', 'eig', 'qr', 'svd-qr', 'svd-jacobi'],
    'fit_intercept': [True, False],
    'normalize': [True, False]
}


