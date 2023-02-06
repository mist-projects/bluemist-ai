import logging
import os
import urllib.parse
from logging import config

import cx_Oracle
import pandas as pd
from sqlalchemy import create_engine

BLUEMIST_PATH = os.getenv("BLUEMIST_PATH")
config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def get_data_from_database(db_type=None, host=None, database=None, service=None, oracle_instant_client_path=None,
                           username=None, password=None, query=None,
                           chunk_size=1000):
    password = urllib.parse.quote_plus(password)

    if db_type == 'mariadb':
        logger.info('Pulling data from MariaDB')
        connection_url = 'mysql+pymysql://' + username + ':' + password + '@' + host + '/' + database
        engine = create_engine(connection_url)
        conn = engine.connect()
        logger.info('Connection successful !!')
        data = extract_data(conn, query, chunk_size)
        return data
    elif db_type == 'mssql':
        logger.info('Pulling data from MS SQL')
        connection_url = 'mssql+pymssql://' + username + ':' + password + '@' + host + '/' + database
        engine = create_engine(connection_url)
        conn = engine.connect()
        logger.info('Connection successful !!')
        data = extract_data(conn, query, chunk_size)
        return data
    elif db_type == 'mysql' or db_type == 'aurora-mysql':
        logger.info('Pulling data from MySQL')
        connection_url = 'mysql+pymysql://' + username + ':' + password + '@' + host + '/' + database
        engine = create_engine(connection_url)
        conn = engine.connect()
        logger.info('Connection successful !!')
        data = extract_data(conn, query, chunk_size)
        return data
    elif db_type == 'oracle':
        logger.info('Pulling data from Oracle')
        connection_url = 'oracle+cx_oracle://' + username + ':' + password + '@' + host + '/?service_name=' + service
        engine = create_engine(connection_url)
        conn = engine.connect()
        logger.info('Connection successful !!')
        data = extract_data(conn, query, chunk_size)
        return data
    elif db_type == 'postgres' or db_type == 'aurora-postgres':
        logger.info('Pulling data from PostgreSQL')
        connection_url = 'postgresql://' + username + ':' + password + '@' + host + '/' + database
        engine = create_engine(connection_url)
        conn = engine.connect()
        logger.info('Connection successful !!')
        data = extract_data(conn, query, chunk_size)
        return data


def extract_data(conn=None, query=None, chunk_size=None):
    dfs = []
    record_count = 0

    for chunk in pd.read_sql_query(sql=query, con=conn, chunksize=chunk_size):
        dfs.append(chunk)
        record_count = record_count + chunk.shape[0]
        logger.debug('Records pulled in this batch {}'.format(chunk.shape[0]))
    data = pd.concat(dfs, ignore_index=True)
    logger.info('Total records pulled {}'.format(record_count))

    return data
