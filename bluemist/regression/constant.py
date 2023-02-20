
__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


multi_output_regressors = [
    'MultiOutputRegressor',
    'RegressorChain'
]

multi_task_regressors = [
    'MultiTaskElasticNet',
    'MultiTaskLasso',
    'MultiTaskElasticNetCV',
    'MultiTaskLassoCV'
]

unsupported_regressors = [
    'IsotonicRegression'
]

base_estimator_regressors = [
    'StackingRegressor',
    'VotingRegressor'
]
