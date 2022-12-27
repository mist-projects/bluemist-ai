import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from preprocessing import numeric_transformer, categorical_transformer


def preprocess_data(
        data,
        drop_features=None,
        convert_column_datatype=None,
        numerical_features=None,
        categorical_features=None,
        convert_to_nan=None,
        scaler='StandardScaler',
        missing_value=np.nan,
        numeric_imputer_strategy='mean',
        numeric_constant_value=None,
        categorical_imputer_strategy='most_frequent',
        categorical_constant_value=None,
        categorical_encoder='LabelEncoder',
        drop_categories_one_hot_encoder=None,
        handle_unknown_one_hot_encoder=None):
    """
    data: dataframe = None
        Dataframe to be passed to the ML estimator
    drop_features: str or list, default=None
        List of features to be dropped from the dataset
    convert_column_datatype: str ot list
        Convert data type of column to another data type
    numerical_features: list, default=None
        list of numerical features
    categorical_features: list, default=None
        list of categorical features
    convert_to_nan: str, list, dict, Series, int, float, or None
        dataset vales to be converted NaN
    missing_value: object, defaulr=np.nan
        value to be imputed
    imputer_strategy: str, default='mean'
        imputation strategy. Possible values are 'mean, 'median', 'most_frequent', 'constant'
    constant_value: str or number, default = None
        constant_value will replace the missing_value when imputer_strategy is passed as 'constant'
    """

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

    # Also replacing the categorical var with actual values
    # data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
    # data = pd.get_dummies(data, columns=['origin'])

    # isdigit()? on 'horsepower'
    hpIsDigit = pd.DataFrame(
        data.horsepower.str.isdigit())  # if the string is made of digits store True else False

    # print isDigit = False!
    data[hpIsDigit['horsepower'] == False]  # from temp take only those rows where hp has false

    data = data.replace('?', np.nan)
    data[hpIsDigit['horsepower'] == False]

    medianFiller = lambda x: x.fillna(x.median())
    data = data.apply(medianFiller, axis=0)

    data['horsepower'] = data['horsepower'].astype(
        'float64')  # converting the hp column from object / string type to float

    #data = data.replace(convert_to_nan, np.nan)

    num_transformer = numeric_transformer.build_numeric_transformer_pipeline(**locals())
    cat_transformer = categorical_transformer.build_categorical_transformer_pipeline(**locals())

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric_transformer", num_transformer, final_numerical_features),
            ("categorical_transformer", categorical_transformer, final_categorical_features)
        ]
    )
    return data
