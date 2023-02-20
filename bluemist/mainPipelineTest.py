
__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


import joblib
import pandas as pd

preprocessor = joblib.load('artifacts/preprocessor/preprocessor.joblib')
pipeline = joblib.load('artifacts/models/ARDRegression.joblib')

def main():
    data = {
        'cylinders': [8],
        'displacement': [307],
        'horsepower': [130],
        'weight': [3504],
        'acceleration': [12],
        'model_year': [70],
        'origin': '1'
    }

    input_df = pd.DataFrame(data, columns=[
        'cylinders',
        'displacement',
        'horsepower',
        'weight',
        'acceleration',
        'model_year',
        'origin',
        ])

    df_to_predict = pd.DataFrame(preprocessor.transform(input_df), columns=[
        'cylinders',
        'displacement',
        'horsepower',
        'weight',
        'acceleration',
        'model_year',
        'origin_1',
        'origin_2',
        'origin_3',
        ])

    # Predicting the Class
    prediction = pipeline.predict(df_to_predict)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
