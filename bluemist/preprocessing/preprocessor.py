"""
Performs data pre-processing
"""

__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


import logging
import os
from logging import config
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from bluemist.pipeline.bluemist_pipeline import save_preprocessor
from bluemist.preprocessing import categorical_transformer, numeric_transformer

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")

initial_column_metadata_for_deployment = []
encoded_columns_for_deployment = []
target_for_deployment = None


def preprocess_data(
        data,
        target_variable,
        test_size=0.25,
        data_randomizer=None,
        drop_features=None,
        numerical_features=None,
        force_numeric_conversion=True,
        categorical_features=None,
        convert_values_to_nan=None,
        data_scaling_strategy='StandardScaler',
        data_tranformation_strategy=None,
        missing_values=np.nan,
        numeric_imputer_strategy='mean',
        numeric_constant_value=None,
        categorical_imputer_strategy='most_frequent',
        categorical_constant_value=None,
        categorical_encoder='OneHotEncoder',
        drop_categories_one_hot_encoder=None,
        handle_unknown_one_hot_encoder=None):
    """
        data: pandas daframe
            Dataframe to be processed before passing to the ML estimator
        target_variable: str
            Target variable to be predicted
        test_size: float or int, default=0.25
            Percentage of the data to be used for testing model performance
        data_randomizer: int default=None
            Controls the data split. Provide a value to reproduce the same split.
        drop_features: str ot list
            Drops the features from the dataset
        numerical_features: list, default=None
            Bluemist AI will automatically identify numerical features from the dataset. Provide the list of features to override the type identified by Bluemist AI.
        force_numeric_conversion: bool, default=True
            Gracefully converts features to numeric datatype which are provided under ``numerical_features``
        categorical_features: list, default=None
            Bluemist AI will automatically identify categorical features from the dataset. Provide the list of features to override the type identified by Bluemist.
        convert_to_nan:  str, list, default=None
            Dataset values to be converted to NumPy NaN
        data_scaling_strategy: {None, 'StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'}, default='StandardScaler'
            Scales dataset features, excluding target variable
                - 'StandardScaler':
                - 'MinMaxScaler'
                - 'MaxAbsScaler'
                - 'RobustScaler'
        data_tranformation_strategy: {'box-cox', 'yeo-johnson' or None}, default=None
            Transforms the features, excluding target variable.
                - 'box-cox':
                - 'yeo-johnson':
        missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan
            All instances of missing_value will be replaced with the user provided imputer strategy
        numeric_imputer_strategy: {'mean, 'median', 'most_frequent', 'constant'}, default='mean'
            Replaces `missing_values` with the strategy provided
        numeric_constant_value: str or number, default=None
            ``numeric_constant_value`` will replace the ``missing_values`` when ``numeric_imputer_strategy`` is passed as ``constant``
        categorical_imputer_strategy:  {'most_frequent', 'constant'}, default='most_frequent'
            Replaces `missing_values` with the strategy provided
        categorical_constant_value: str or number, default=None
            ``categorical_constant_value`` will replace the ``missing_values`` when ``categorical_imputer_strategy`` is passed as ``constant``
        categorical_encoder: {'OneHotEncoder', 'OrdinalEncoder'}, default='OneHotEncoder'
            Encode categorical features
        drop_categories_one_hot_encoder: {‘first’, ‘if_binary’ or None}, default='None'
            Determines strategy to drop one category per feature
                - 'first':
                    drops the first category for each feature.
                - 'if_binary':
                    drops the first category for features with teo categories
                - None:
                    Keeps all features and categories
        handle_unknown_one_hot_encoder : {‘error’, ‘ignore’, ‘infrequent_if_exist’}, default=’error’
            Handles unknown category during transform
                - 'error':
                    throws an error if category is unknown
                - 'ignore':
                    ignores if category is unknown, output encoded column for this feature will be all zeroes
                - 'infrequent_if_exist':
                    unknown category will be mapped to infrequent category if exists. If infrequent category does not exist, it
                    will be treated as `ignore`
    """

    global target_for_deployment
    target_for_deployment = target_variable

    logger.info('Shape of the dataset :: {}'.format(data.shape))
    logger.info('Columns in the dataset :: \n{}'.format(data.columns))

    # drop features from the dataset
    if drop_features is not None:
        if isinstance(drop_features, str):
            data.drop([drop_features], axis=1, inplace=True)
        elif isinstance(drop_features, list):
            data.drop(drop_features, axis=1, inplace=True)

    # auto compute numerical and categorical features
    auto_computed_numerical_features = data.select_dtypes(include='number').columns.tolist()
    auto_computed_categorical_features = data.select_dtypes(include='object').columns.tolist()

    final_numerical_features = auto_computed_numerical_features.copy()
    final_numerical_features.remove(target_variable)
    final_categorical_features = auto_computed_categorical_features.copy()

    # finalize the list of numerical features
    if auto_computed_numerical_features is not None:
        if numerical_features is not None:
            for numerical_feature in numerical_features:
                if numerical_feature not in auto_computed_numerical_features:
                    final_numerical_features.append(numerical_feature)

        if categorical_features is not None:
            for categorical_feature in categorical_features:
                if categorical_feature in auto_computed_numerical_features:
                    final_numerical_features.remove(categorical_feature)

    # finalize the list of categorical features
    if auto_computed_categorical_features is not None:
        if categorical_features is not None:
            for categorical_feature in categorical_features:
                if categorical_feature not in auto_computed_categorical_features:
                    final_categorical_features.append(categorical_feature)

        if numerical_features is not None:
            for numerical_feature in numerical_features:
                if numerical_feature in auto_computed_categorical_features:
                    final_categorical_features.remove(numerical_feature)

    # prepare final list of columns after preprocessing
    column_list = []
    if bool(final_numerical_features) and bool(final_categorical_features):
        column_list = final_numerical_features + final_categorical_features
    elif bool(final_numerical_features):
        column_list = final_numerical_features
    elif bool(final_categorical_features):
        column_list = final_categorical_features

    # handle non-numeric data in user provided numeric column
    if numerical_features is not None:
        if force_numeric_conversion:
            numeric_conversion_strategy = 'coerce'
        else:
            numeric_conversion_strategy = 'raise'

    logger.debug('data.dtypes before preprocessing  :: \n{}'.format(data.dtypes))

    if numerical_features is not None:
        data[final_numerical_features] = data[final_numerical_features].apply(pd.to_numeric, errors=numeric_conversion_strategy, axis=1)

    data[final_categorical_features] = data[final_categorical_features].astype(str)
    logger.debug('data.dtypes after dtype conversion  :: \n{}'.format(data.dtypes))

    # Creating list of column name and datatype which will be used in generate_api.py
    global initial_column_metadata_for_deployment
    for col_name, col_type in data.drop(target_variable, axis=1).dtypes.items():
        initial_column_metadata_for_deployment.append((col_name, col_type))

    # create transformers for preprocessing pipeline
    num_transformer = numeric_transformer.build_numeric_transformer_pipeline(**locals())
    cat_transformer = categorical_transformer.build_categorical_transformer_pipeline(**locals())

    # create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric_transformer", num_transformer, final_numerical_features),
            ("categorical_transformer", cat_transformer, final_categorical_features)
        ], verbose_feature_names_out=False
    )

    X = data.drop([target_variable], axis=1)
    y = data[[target_variable]]
    logger.debug('Splitting dataset into X_train, X_test, y_train, y_test...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=data_randomizer)

    logger.debug('X_train.dtypes before ColumnTransformer :: \n{}'.format(X_train.dtypes))
    X_train = pd.DataFrame(preprocessor.fit_transform(X_train), columns=preprocessor.get_feature_names_out())
    X_test = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())

    global encoded_columns_for_deployment
    encoded_columns_for_deployment = preprocessor.get_feature_names_out()

    logger.debug('X_train Columns after ColumnTransformer :: {}'.format(preprocessor.get_feature_names_out()))
    logger.debug('X_train.dtypes after ColumnTransformer :: \n{}'.format(X_train.dtypes))

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    logger.debug('Saving preprocessor to disk...')
    save_preprocessor(preprocessor)
    return X_train, X_test, y_train, y_test
