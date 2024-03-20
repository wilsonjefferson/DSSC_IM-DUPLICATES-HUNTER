"""
    This script contains the FileHandler class to handle
    a (excel) file.

    @file: FileHandler.py
    @version: v. 1.1
    @last update: 16/06/2022
    @author: Pietro Morichetti
"""

import pandas as pd
import os

os.environ["HTTP_PROXY"] = "http://10.141.0.165:3128"
os.environ["HTTPS_PROXY"]= "http://10.141.0.165:3128"
os.environ["NO_PROXY"] = "localhost, 127.0.0.1, *.azureml.net"

import logging

log = logging.getLogger('sourceHandlerLogger')

from exceptions.FileNotExistWarning import FileNotExistWarning
from exceptions.NoValidExtensionWarning import NoValidExtensionWarning
from exceptions.SourceNotValidException import SourceNotValidException
from exceptions.DISCQueryNotValidException import DISCQueryNotValidException
from core.data_manipulation.data_handlers.QueryHandler import QueryHandler

Query = QueryHandler.Query


def unix_connector_is_table(disc:object, source:str) -> bool:
    query_result = disc._cursor.execute(f"DESCRIBE FORMATTED {source}").fetchall()
    table_type = [r for r in query_result  if 'Table Type' in (str(s).strip().replace(':', '') for s in r)][0][1].strip().replace(':', '')
    return table_type == 'MANAGED_TABLE'


class SourceHandler:

    def __init__(self):
        self.source = None
        self.db_pltf = None

    def __del__(self):
        if self.db_pltf:
            self.db_pltf.close()
            del self.db_pltf

    def connect(self, db_platform:str, source:str) -> bool:
        '''
            Check if the table exist in database.

            Parameters
            ----------
            source: str
                location of the table, it should be something like: <database_name>.<table_name>

            Returns
            -------
            True if the table is present in the database, False otherwise
        '''
        
        import platform
        self.pltf = platform.system()
        log.info(f'current os platform: {self.pltf}')

        self._set_db_platform(db_platform)
        self.db_pltf.close()

        log.info('connecting to DB Platform...')
        self.db_pltf.connect(dsn=db_platform)
        log.info(self.db_pltf)
        
        db, table = source.split('.')
        
        log.debug('check remote DB and table existency')

        if self.pltf == 'Linux' or self.pltf == 'Mac':
            exist = unix_connector_is_table(self.db_pltf, source)
        elif self.pltf == 'Windows':
            exist = self.db_pltf.is_table(table, db)
        else:
            log.info('OS is not recognized, running through pyodbc...')
            import pyodbc

            self.db_pltf = pyodbc.connect(f'DSN={db_platform}')
            cnxn = self.db_pltf.cursor()

            cnxn.execute(f'SHOW TABLES IN {db}')
            exist = [True for i in cnxn.fetchall() if table in i]
            exist = True if exist else False    
        
        if exist is False:
            FileNotExistWarning(source)
            return False
        
        self.source = source
        return True

    def _set_db_platform(self, db_platform:str, query:str = None) -> pd.DataFrame:
        '''
            Retrive the source data from the location provided.

            Parameters
            ----------
            source: str
                location of the source

            Returns
            -------
            pd.DataFrame the retrieved data as a Dataframe
        '''

        from connectors import disc, devo  # [ECB library] Opens connection to DISC

        if db_platform == 'DISC':
            self.db_pltf = disc
        elif db_platform == 'DEVO':
            self.db_pltf = devo
        else:
            raise SourceNotValidException(db_platform)

    def get_data(self, query_command:str = None) -> pd.DataFrame:
        
        query = self._get_query(query_command)
        query = query.format(self.source)

        df = None
        log.debug('query: %s'%query)

        log.debug('Read data from remote DISC')
        df = self.db_pltf.read_sql(query)
        
        if len(df) == 0:
            log.warning('no record retrieved, target dataframe empty.\nprogram ending...')
            exit()

        if query_command == 'RETRIEVE_YESTERDAY':
            last_time = df.opentime.iat[-1]
            log.info(f'last incident was raised {last_time}')

        return df

    def _get_query(self, query:str = None) -> str:
        '''
            Get the absolute path of the source file 
            and check if it is valid.

            Parameters
            ----------
            None

            Returns
            -------
                pd.DataFrame
                Dataframe of incidents
        '''

        if query == 'RETRIEVE_YESTERDAY':
            from ConfigNameSpace import _config_name_space, last_time, YESTERDAY
            last_time = YESTERDAY if last_time == 'none' else last_time
            log.info(f'retrieve incidents raised after {last_time}')

            # store last_time variable in config.yaml file
            _config_name_space.constants.GLOBAL.last_incident_time = str(last_time)
            _config_name_space.store()
            # NOTE: not elegant but imported last_time variable does not overwrite
            # _config_name_space yaml attribute so it is necessary to force it
            # by directly overwrite from the _config_name_space object.
            #---------------------------------------------

            query = Query.RETRIEVE_YESTERDAY.format('{}', last_time)
        elif query == 'RETRIEVE_LAST_2_WEEKS':
            query = Query.RETRIEVE_LAST_2_WEEKS
        elif query == 'RETRIEVE_ALL' or query is None:
            query = Query.RETRIEVE_ALL
        else:
            raise DISCQueryNotValidException(query, Query)

        return query

    def store(self, df:pd.DataFrame) -> None:
        vals = df.apply(lambda x: ['\'' + i + '\'' if type(i)==str else str(i) for i in x], axis=1)
        vals = vals.apply(lambda x: ','.join(x))
        vals = list(map(lambda x: '(' + x + ')', vals))          
        vals = ','.join(vals)
        
        query = Query.INSERT.format(self.source, vals)
        log.debug(query)

        self.db_pltf._cursor.execute(query)