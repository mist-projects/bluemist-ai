from datasource.file import get_csv_data


def get_data(file_type='csv'):
    if file_type == 'csv':
        data = get_csv_data()
        return data
