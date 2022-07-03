import joblib
import pandas as pd


def main():
    pipeline = joblib.load('api/models/LinearRegression.joblib')
    print(pipeline)
    data = {
        'cylinders': [8],
        'displacement': [307],
        'horsepower': [130],
        'weight': [3504],
        'acceleration': [12],
        'model year': [70]
    }
    df_to_predict = pd.DataFrame.from_dict(data)
    prediction = pipeline.predict(df_to_predict)
    print('prediction', prediction)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
