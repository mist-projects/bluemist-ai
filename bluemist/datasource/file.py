import logging
import os
from logging import config
import pandas as pd


BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")
config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def get_data_from_filesystem(file_path, sheet_name=0, file_type='csv', separator=',', delimiter=None):
    logger.info('Pulling data from {}'.format(file_path))

    if file_type == 'csv':
        data = pd.read_csv(filepath_or_buffer=file_path, sep=separator, delimiter=delimiter)
        logger.info('Data pull completed !!')
        return data
    elif file_type == 'excel':
        data = pd.read_excel(io=file_path, sheet_name=sheet_name)
        logger.info('Data pull completed !!')
        return data


