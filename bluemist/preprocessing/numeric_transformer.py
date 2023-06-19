# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.2
# Email: dew@bluemist-ai.one
# Created: Sep 6, 2022
# Last modified: June 19, 2023


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from bluemist.utils.scaler import get_scaler, available_scalers


def build_numeric_transformer_pipeline(**kwargs):
    preprocessing_steps = []

    imputer_strategy = kwargs.get('numeric_imputer_strategy')
    scaler = kwargs.get('data_scaling_strategy')
    numeric_constant_value = kwargs.get('numeric_constant_value')
    data_tranformation_strategy = kwargs.get('data_tranformation_strategy')

    # handling missing values
    if imputer_strategy is not None and imputer_strategy in ['mean', 'median', 'most_frequent', 'constant']:
        if imputer_strategy == 'constant':
            imputer_step = ('imputer', SimpleImputer(strategy=imputer_strategy, fill_value=numeric_constant_value))
        else:
            imputer_step = ('imputer', SimpleImputer(strategy=imputer_strategy))
        preprocessing_steps.append(imputer_step)
    elif imputer_strategy is not None:
        raise ValueError('Invalid imputer_strategy value passed : ', imputer_strategy)

    # data scaling
    if scaler is not None and scaler in available_scalers:
        scaler_step = ('scaler', get_scaler(scaler))
        preprocessing_steps.append(scaler_step)
    elif scaler is not None:
        raise ValueError('Invalid scaler value passed : ', scaler)

    # data transformation
    if data_tranformation_strategy is not None and data_tranformation_strategy in ['auto', 'yeo-johnson', 'box-cox']:
        transformer_step = tuple()
        if data_tranformation_strategy == 'yeo-johnson':
            transformer_step = ('transformer', PowerTransformer(method='yeo-johnson'))
        elif data_tranformation_strategy == 'box-cox':
            transformer_step = ('transformer', PowerTransformer(method='box-cox'))
        preprocessing_steps.append(transformer_step)

    numeric_transformer = Pipeline(steps=preprocessing_steps)
    return numeric_transformer
