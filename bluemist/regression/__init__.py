
__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


from bluemist.regression.regressor import (
    get_estimators,
    train_test_evaluate,
    deploy_model
)

__all__ = ['get_estimators',
           'train_test_evaluate',
           'deploy_model'
]