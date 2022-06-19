# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.linear_model import LinearRegression

import regression.regression
import utils.datahandler
from utils.datahandler import extractData
from sklearn.model_selection import train_test_split  # Sklearn package's randomized data splitting function


def main():
    # Use a breakpoint in the code line below to debug your script.
    data_handler = utils.datahandler.extractData()
    data = data_handler.get_data()
    print(data.shape)

    X = data.drop(['mpg', 'origin_europe'], axis=1)
    # the dependent variable
    y = data[['mpg']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    regression.regression.regressor.train_model(X_train, X_test, y_train, y_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

    from sklearn.utils import all_estimators

    estimators = all_estimators(type_filter='regressor')

    all_regs = []
    for name, RegressorClass in estimators:
        try:
            print('Appending name', name)
            print('Appending class', RegressorClass)
            reg = RegressorClass()
            print(reg.get_params())
            all_regs.append(reg)
        except Exception as e:
            print(e)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
