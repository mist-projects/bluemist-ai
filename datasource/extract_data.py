from datasource.provider.aws import get_data_from_s3
from datasource.provider.file import get_csv_data


def get_data(file_type='csv', file_path='', provider='', access_key='', access_secret=''):
    if file_type == 'csv':
        data = get_csv_data()
        return data

