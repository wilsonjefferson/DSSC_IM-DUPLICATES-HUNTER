from dataclasses import dataclass
from datetime import datetime, timedelta


date_format = '%Y-%m-%d'

class QueryHandler:
    '''
        This is a container class to handle SQL query for DISC.
        
    '''

    @dataclass(frozen=True)
    class Query:
        '''
            This is a dataclass to store static or templetized SQL queries to
            retrieve tickets from DISC.

            Attributes
            ----------

            DATA: str
                Initial part of a SQL query with the list of items to retrieve

            RETRIEVE_ALL: str
                Retrieve all the IM tickets from 2019-01-01

            RETRIEVE_TODAY: str
                Retrieve the IM tickets opened today
        
            RETRIEVE_LAST_2_WEEKS: str
                Retrieve IM tickets during the last 2 weeks
        '''

        SELECT: str = """SELECT incidentid, opentime, resolvedtime, title, description, 
                assignmentgroup, assignee, solution, last_resolved_by, status, 
                requester, primaryaffectedservice, priority, previousassignmentgroup, 
                affectedci, closedby, category, subcategory, area, reopentime"""

        RETRIEVE_ALL: str = SELECT + """
                FROM {} 
                WHERE opentime >= '2019-01-01 00:00:00.0'
                AND opentime < '""" + datetime.today().strftime(date_format) + """ 00:00:00.0'
                AND status IN ('Closed', 'Resolved')
                ORDER BY opentime"""

        # DISC/DEVO received data 24 hours later, so for the today incidents you
        # need to wait tomorrow morning. For that reason RETRIEVE_YESTERDAY stops
        # the day before today (yesterday)
        RETRIEVE_YESTERDAY: str = SELECT + """
                FROM {} 
                WHERE opentime > '{} 00:00:00.0'
                AND status NOT IN ('Closed', 'Resolved')
                ORDER BY opentime"""

        # DISC/DEVO received data 24 hours later, so for the today incidents you
        # need to wait tomorrow morning. For that reason RETRIEVE_LAST_2_WEEKS stops
        # the day before today (yesterday)
        RETRIEVE_LAST_2_WEEKS: str = SELECT + """
                FROM {} 
                WHERE opentime < '""" + (datetime.today() - timedelta(days = 1)).strftime(date_format) + """ 00:00:00.0'
                AND opentime >= '""" + (datetime.today() - timedelta(days = 14)).strftime(date_format) + """ 00:00:00.0'
                ORDER BY opentime"""
        
        INSERT: str = """INSERT INTO {} VALUES {}"""
