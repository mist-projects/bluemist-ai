# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.2
# Email: dew@bluemist-ai.one
# Created: Sep 6, 2022
# Last modified: May 29, 2023

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

from bluemist import environment


def build_categorical_transformer_pipeline(**kwargs):
    transformer_steps = []

    imputer_strategy = kwargs.get('categorical_imputer_strategy')
    categorical_constant_value = kwargs.get('categorical_constant_value')
    categorical_encoder = kwargs.get('categorical_encoder')
    drop_categories = kwargs.get('drop_categories_one_hot_encoder')
    handle_unknown = kwargs.get('handle_unknown_one_hot_encoder')

    if imputer_strategy is not None and imputer_strategy in ['most_frequent', 'constant']:
        if imputer_strategy == 'constant':
            imputer_step = ('imputer', SimpleImputer(strategy=imputer_strategy, fill_value=categorical_constant_value))
        else:
            imputer_step = ('imputer', SimpleImputer(strategy=imputer_strategy))
        transformer_steps.append(imputer_step)
    else:
        raise ValueError('Invalid imputer_strategy value passed: {}'.format(imputer_strategy))

    if categorical_encoder is not None and categorical_encoder in ['OneHotEncoder', 'OrdinalEncoder']:
        encoder_step = tuple()
        if categorical_encoder == 'LabelEncoder':
            encoder_step = ('label_encoder', LabelEncoder())
        elif categorical_encoder == 'OrdinalEncoder':
            encoder_step = ('ordinal_encoder', OrdinalEncoder())
        elif categorical_encoder == 'OneHotEncoder':
            encoder_class = OneHotEncoder
            kwargs = {}
            if drop_categories is not None:
                kwargs['drop'] = drop_categories
            if handle_unknown is not None:
                kwargs['handle_unknown'] = handle_unknown

            encoder_step = ('one_hot_encoder', encoder_class(**kwargs))

        transformer_steps.append(encoder_step)
    else:
        raise ValueError('Invalid categorical_encoder value passed : ', categorical_encoder)

    categorical_transformer = Pipeline(steps=transformer_steps)
    return categorical_transformer
