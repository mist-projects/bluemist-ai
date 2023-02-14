"""
Extract data from local file system
"""

__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


import logging
import os
from logging import config
import pandas as pd

BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def get_data_from_filesystem(file_path,
                             file_type='delimited',
                             sheet_name=0,
                             delimiter=','):
    """
        file_path: str
            File system path of the data file to be extracted
        file_type: {'delimited', 'excel'}, default='delimited'
            Type of the data file
        sheet_name: str, default=0
            Sheet name if ``file_type`` is ``excel``
        delimiter: str, default=','
            File delimiter to use if ``file_type`` is ``delimited``
    """

    logger.info('Pulling data from {}'.format(file_path))

    if file_type == 'delimited':
        data = pd.read_csv(filepath_or_buffer=file_path, sep=delimiter)
        logger.info('Data pull completed !!')
        return data
    elif file_type == 'excel':
        data = pd.read_excel(io=file_path, sheet_name=sheet_name)
        logger.info('Data pull completed !!')
        return data


