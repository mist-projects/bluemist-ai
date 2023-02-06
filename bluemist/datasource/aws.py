import logging
import os
from logging import config
import pandas as pd
import boto3

BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")
config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def get_data_from_s3(aws_access_key_id, aws_secret_access_key, s3_bucket_name, s3_object_name, destination_path):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    logger.debug('S3 service client created :: {}'.format(s3))
    s3.download_file(s3_bucket_name, s3_object_name, destination_path)
    data = pd.read_csv(destination_path)
    return data
