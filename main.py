from datasource.file import get_data_from_filesystem
from preprocessing import preprocess_data
from regression import train_test_evaluate


def main():

    data = get_data_from_filesystem('datasets/auto-mpg/auto-mpg.csv')

    print(data.shape)
    data = preprocess_data(data)
    train_test_evaluate(data, tune_models=None, metrics='all')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
