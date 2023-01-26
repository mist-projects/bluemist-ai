import joblib
import pandas as pd


def main():
    preprocessor = joblib.load('artifcats/preprocessor/preprocessor.joblib')
    print(preprocessor)
    pipeline = joblib.load('artifcats/models/LinearRegression.joblib')
    print(pipeline)
    data = {
        'cylinders': [8],
        'displacement': [307],
        'weight': [3504],
        'acceleration': [12],
        'model_year': [70],
        'horsepower': [130]
    }
    df_to_predict = pd.DataFrame.from_dict(data)

    df_to_predict = pd.DataFrame(preprocessor.transform(df_to_predict), columns=['cylinders', 'displacement', 'weight',
                                 'acceleration', 'model_year', 'horsepower'])
    print(df_to_predict)
    prediction = pipeline.predict(df_to_predict)
    print('prediction', prediction)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
