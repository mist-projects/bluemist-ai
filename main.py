import sys

from pyfiglet import Figlet
from termcolor import colored

from datasource.database import get_data_from_database
from datasource.file import get_data_from_filesystem
from preprocessing import preprocess_data
from regression import train_test_evaluate


def main():
    f = Figlet(font='small')
    print(colored(f.renderText('B l u e    M i s t - AI'), 'blue'))

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

    print('sys.platform.', sys.platform)
    # query = 'SELECT * FROM AUTO_MPG'
    # data = get_data_from_database(db_type='oracle', host='oracle.cxh6nuaszc34.us-east-1.rds.amazonaws.com',
    #                               username='admin', password='adminadmin', service='DATABASE',
    #                               oracle_instant_client_path='/home/shashank-agrawal/Desktop/instantclient_21_6',
    #                               query=query, chunk_size=100)

    data = get_data_from_filesystem('datasets/auto-mpg/auto-mpg.csv')
    print(data.shape)
    data = preprocess_data(data, drop_features=['car name', 'origin'])
    train_test_evaluate(data, tune_models='all', metrics='all', mlflow_stats=False, scale_data=True, scale_target=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
