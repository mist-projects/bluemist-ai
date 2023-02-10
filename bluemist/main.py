import logging
import os
import sys
from logging import config

import bluemist
from bluemist.environment import initialize
from bluemist.datasource.aws import get_data_from_s3
from bluemist.datasource import get_data_from_filesystem
from bluemist.pipeline import get_model_pipeline
from bluemist.preprocessing import preprocess_data
from bluemist.regression import train_test_evaluate, deploy_model, get_estimators

HOME_PATH = os.environ["BLUEMIST_PATH"]
config.fileConfig(HOME_PATH + '/' + 'logging.config')
logger = logging.getLogger("root")


def main():
    # query = 'SELECT * FROM public.auto_mpg'
    # data = get_data_from_database(db_type='postgres', host='postgres.cxh6nuaszc34.us-east-1.rds.amazonaws.com:5432',
    #                        username='postgres', password='adminadmin', database='postgres', query=query, chunk_size=100)

    # query = 'SELECT * FROM public.auto_mpg'
    # data = get_data_from_database(db_type='aurora-postgres', host='aurora-postgres-instance-1.cxh6nuaszc34.us-east-1.rds.amazonaws.com:5432',
    #                        username='postgres', password='adminadmin', database='postgres', query=query, chunk_size=100)

    # query = 'SELECT * FROM auto_mpg'
    # data = get_data_from_database(db_type='mysql', host='mysql.cxh6nuaszc34.us-east-1.rds.amazonaws.com:3306',
    #                        username='admin', password='adminadmin', database='sys', query=query, chunk_size=100)

    # query = 'SELECT * FROM AUTO_MPG'
    # data = get_data_from_database(db_type='aurora-mysql', host='aurora-mysql-instance-1.cxh6nuaszc34.us-east-1.rds.amazonaws.com:3306',
    #                        username='admin', password='adminadmin', database='sys', query=query, chunk_size=100)

    # query = 'SELECT * FROM dbo.auto_mpg'
    # data = get_data_from_database(db_type='mssql', host='mssql.cxh6nuaszc34.us-east-1.rds.amazonaws.com:1433',
    #                        username='admin', password='adminadmin', database='bluemist', query=query, chunk_size=100)

    # query = 'SELECT * FROM auto_mpg'
    # data = get_data_from_database(db_type='mariadb', host='mariadb.cxh6nuaszc34.us-east-1.rds.amazonaws.com:3306',
    #                        username='admin', password='adminadmin', database='innodb', query=query, chunk_size=100)

     # query = 'SELECT * FROM AUTO_MPG'
    # data = get_data_from_database(db_type='oracle', host='oracle.cxh6nuaszc34.us-east-1.rds.amazonaws.com',
    #                               username='admin', password='adminadmin', service='DATABASE',
    #                               oracle_instant_client_path='/home/shashank-agrawal/Desktop/instantclient_21_6',
    #                               query=query, chunk_size=100)

    #print('get_estimators', get_estimators())
    initialize()
    data = get_data_from_filesystem('datasets/auto-mpg/auto-mpg.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data, target_variable='mpg', test_size=0.25,
                                                       drop_features=['car name'],
                                                       numerical_features=['horsepower'],
                                                       categorical_features=['origin'],
                                                       categorical_encoder='OneHotEncoder')
    #
    # train_test_evaluate(X_train, X_test, y_train, y_test, tune_models=None, metrics='all', target_scaling_strategy='MinMaxScaler')
    # # pipeline = get_model_pipeline('LarsCV')
    # # print(pipeline.get_params)
    # #deploy_model(estimator_name='LarsCV', host='localhost', port=8000)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
