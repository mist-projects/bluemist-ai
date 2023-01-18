import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


def build_numeric_transformer_pipeline(**kwargs):
    """
    :rtype: object
    """
    transformer_steps = []

    imputer_strategy = kwargs.get('numeric_imputer_strategy')
    scaler = kwargs.get('scaler')
    numeric_constant_value = kwargs.get('numeric_constant_value')

    if imputer_strategy is not None and imputer_strategy in ['mean', 'median', 'most_frequent', 'constant']:
        if imputer_strategy == 'constant':
            imputer_step = ('imputer', SimpleImputer(strategy=imputer_strategy, fill_value=numeric_constant_value))
        else:
            imputer_step = ('imputer', SimpleImputer(strategy=imputer_strategy))
        transformer_steps.append(imputer_step)
    else:
        raise ValueError('Invalid imputer_strategy value passed : ', imputer_strategy)

    # if scaler is not None and scaler in ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler']:
    #     if scaler == 'StandardScaler':
    #         scaler_step = ('scaler', StandardScaler())
    #     elif scaler == 'MinMaxScaler':
    #         scaler_step = ('scaler', MinMaxScaler())
    #     elif scaler == 'MaxAbsScaler':
    #         scaler_step = ('scaler', MaxAbsScaler())
    #     elif scaler == 'RobustScaler':
    #         scaler_step = ('scaler', RobustScaler())
    #     transformer_steps.append(scaler_step)
    # else:
    #     raise ValueError('Invalid scaler value passed : ', scaler)

    numeric_transformer = Pipeline(steps=transformer_steps)
    return numeric_transformer
