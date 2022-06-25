from datasource.provider.file import get_csv_data


def get_data(file_type='csv', file_path='', provider=''):
    if file_type == 'csv':
        data = get_csv_data()
        return data
