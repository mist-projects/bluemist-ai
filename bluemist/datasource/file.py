import pandas as pd


def get_data_from_filesystem(file_path, sheet_name=0, file_type='csv', separator=',', delimiter=None):
    if file_type == 'csv':
        data = pd.read_csv(filepath_or_buffer=file_path, sep=separator, delimiter=delimiter)
        return data
    elif file_type == 'excel':
        data = pd.read_excel(io=file_path, sheet_name=sheet_name)
        return data


