import pandas as pd

import boto3


def get_data_from_s3(aws_access_key_id, aws_secret_access_key, s3_bucket_name, s3_object_name, destination_path):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    print('client created', s3)
    s3.download_file(s3_bucket_name, s3_object_name, destination_path)
    data = pd.read_csv(destination_path)
    return data
