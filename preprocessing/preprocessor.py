import numpy as np
import pandas as pd


def preprocess_data(data):
    data = data.drop('car name', axis=1)
    # Also replacing the categorical var with actual values
    data['origin'] = data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
    data = pd.get_dummies(data, columns=['origin'])

    # isdigit()? on 'horsepower'
    hpIsDigit = pd.DataFrame(
        data.horsepower.str.isdigit())  # if the string is made of digits store True else False

    # print isDigit = False!
    data[hpIsDigit['horsepower'] == False]  # from temp take only those rows where hp has false

    data = data.replace('?', np.nan)
    data[hpIsDigit['horsepower'] == False]

    medianFiller = lambda x: x.fillna(x.median())
    data = data.apply(medianFiller, axis=0)

    data['horsepower'] = data['horsepower'].astype('float64')  # converting the hp column from object / string type to float

    return data
