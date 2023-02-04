import os
import urllib.parse

import cx_Oracle
import pandas as pd
from sqlalchemy import create_engine


def get_data_from_database(db_type=None, host=None, database=None, service=None, oracle_instant_client_path=None,
                           username=None, password=None, query=None,
                           chunk_size=1000):
    password = urllib.parse.quote_plus(password)

    if db_type == 'mariadb':
        connection_url = 'mysql+pymysql://' + username + ':' + password + '@' + host + '/' + database
        engine = create_engine(connection_url)
        conn = engine.connect()
        data = extract_data(conn, query, chunk_size)
        return data
    elif db_type == 'mssql':
        connection_url = 'mssql+pymssql://' + username + ':' + password + '@' + host + '/' + database
        engine = create_engine(connection_url)
        conn = engine.connect()
        data = extract_data(conn, query, chunk_size)
        return data
    elif db_type == 'mysql' or db_type == 'aurora-mysql':
        connection_url = 'mysql+pymysql://' + username + ':' + password + '@' + host + '/' + database
        engine = create_engine(connection_url)
        conn = engine.connect()
        data = extract_data(conn, query, chunk_size)
        return data
    elif db_type == 'oracle':
        connection_url = 'oracle+cx_oracle://' + username + ':' + password + '@' + host + '/?service_name=' + service
        engine = create_engine(connection_url)
        conn = engine.connect()
        data = extract_data(conn, query, chunk_size)
        return data
    elif db_type == 'postgres' or db_type == 'aurora-postgres':
        connection_url = 'postgresql://' + username + ':' + password + '@' + host + '/' + database
        engine = create_engine(connection_url)
        conn = engine.connect()
        data = extract_data(conn, query, chunk_size)
        return data


def extract_data(conn=None, query=None, chunk_size=None):
    dfs = []
    record_count = 0

    for chunk in pd.read_sql_query(sql=query, con=conn, chunksize=chunk_size):
        dfs.append(chunk)
        record_count = record_count + chunk.shape[0]
        print('Records pulled in this batch {}'.format(chunk.shape[0]))
    data = pd.concat(dfs, ignore_index=True)
    print('Total records pulled {}'.format(record_count))

    return data
