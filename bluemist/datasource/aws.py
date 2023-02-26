__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"

import logging
import os
from logging import config

from bluemist.datasource import get_data_from_filesystem

BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")
config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def get_data_from_s3(aws_access_key_id,
                     aws_secret_access_key,
                     s3_bucket_name,
                     s3_object_name,
                     destination_path,
                     file_type='delimited',
                     sheet_name=0,
                     delimiter=','):
    """
        Extract data from Amazon cloud (AWS)

        aws_access_key_id: str
            The access key to use when creating the s3 client
        aws_secret_access_key: str
            The secret key to use when creating the s3 client
        s3_bucket_name: str
            s3 bucket name from where the dat file needs to be pulled
        s3_object_name: str
            Name of the data file
        destination_path: str
            File system path where the file will be downloaded from S3
        file_type: {'delimited', 'excel'}, default='delimited'
            type of the data file
        sheet_name: str, default=0
            sheet name if the dataset is an Excel file
        delimiter: str, default=','
            file delimiter to use for delimited files

        Examples
        ---------

        .. raw:: html
           :file: ../../code_samples/quickstarts/datasource/ds_aws.html

    """
    import boto3
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    logger.debug('S3 service client created :: {}'.format(s3))
    s3.download_file(s3_bucket_name, s3_object_name, destination_path)
    data = get_data_from_filesystem(destination_path, sheet_name, file_type, delimiter)
    data.columns = data.columns.str.replace('\W', '_')  # TODO: Revisit the code
    return data
