"""
    This script contains a set of methods to check if
    tickets are violating any ITSM contrains.

    @file: itsm_in_tickets_constrains.py
    @version: v. 1.2.1
    @last update: 23/06/2022
    @author: Pietro Morichetti
"""

import pandas as pd
import numpy as np
import logging
import pyodbc
import networkx as nx
import matplotlib.pyplot as plt
from networkx.exception import NetworkXNoCycle
from typing import List, Dict

from core.utils.CompareCodes import CompareCodes
from exceptions.TicketNotExistWarning import TicketNotExistWarning

log = logging.getLogger('itsmConstrainsLogger')


def check_violation_self_pointing_ticket(origin_tickets:List[str], duplicate_ticket:str, 
        duplicates_violation_same_code:Dict[str, str]) -> None:
    '''
        This method check if a ticket is considering itself
        as a duplicate ticket (self-pointing).

        Parameters
        ----------
        origin_tickets: str
            First ticket raised, first ticket of a series of tickets
            referring to the same "topic"

        duplicate_ticket: str
            A  non-first ticket, since was already raised a previous
            ticket on the same "topic"

        duplicates_violation_same_code: dict
            Dictionary to keep track of tickets violating the current constrain

        Returns
        -------
        None
    '''

    if len(origin_tickets) != 0:
        log.warning('duplicate %s is among the origins'%duplicate_ticket)
        log.warning('violation self pointing ticket detected')
        if len(origin_tickets) == 1 and duplicate_ticket.__eq__(*origin_tickets):
            log.warning('duplicate recorded')
            duplicates_violation_same_code['duplicates'].append(duplicate_ticket)
            duplicates_violation_same_code['origins'].append(*origin_tickets)
        
        if duplicate_ticket in origin_tickets:
            log.debug('duplicate removed among the list of origins')
            origin_tickets.remove(duplicate_ticket) # remove self pointing, if any

def check_violation_cycles_in_star(duplicates:nx.DiGraph) -> None:
    '''
        This method check if there are any cycle in a given star.
        If it is the case, it is shown through a plot.

        Note: Cycles are not allowed.

        Parameters
        ----------
        duplicates: nx.DiGraph
            A star, composed by an origin ticket and a set of (duplicate) tickets
            poiting to the origin ticket, directly or undirectly (through another ticket).

        Returns
        -------
        None
    '''
    
    try:
        '''
            If any cycle is present in the current star, the star will be plotted,
            otherwise nx.find_cycle will raise an exception. In this case the exception
            is catched since it means there are no cycles and the star is valid 
            (so we can go on).
        '''
        nx.find_cycle(duplicates, orientation="original")
        nx.draw(duplicates, with_labels = True)
        plt.show()
    except NetworkXNoCycle as e:
        # no need to do anything, just keep going
        _ = None

def check_violation_duplicate_less_origin(compare_codes:CompareCodes, df:pd.DataFrame, 
        origin_ticket:str, duplicate_ticket:str, duplicates_violation_code_number:Dict[str, str]) \
    -> None:
    '''
        This method check if duplicate ticket is poiting 
        to an origin ticket with a ticket code greater
        than the duplicate ticket code itself.

        If this is the case, the pair of tickets (origin and duplicate)
        are violating an ITSM constrain.

        Example:
        Let's consider PM0010 and PM0050, and PM0010 is origin of PM0050.
        Is int_part_of(PM0010) < int_part_of(PM0050) ?
        - Yes --> Okay, duplicate tickets has integer part greater than the origin
        - No --> Error, not accetable according the ITSM constrain

        Parameters
        ----------
        compare_codes: CompareCodes
            Support class to compare ticket codes

        df: pd.DataFrame
            Table of all tickets

        origin_ticket: str
            First ticket raised, first ticket of a series of tickets
            referring to the same "topic"            

        duplicate_ticket: str
            A  non-first ticket, since was already raised a previous
            ticket on the same "topic"

        duplicates_violation_code_number: dict
            Dictionary to keep track of tickets violating the current constrain

        Returns
        -------
        None
    '''

    if compare_codes.greater(origin_ticket, duplicate_ticket):
        # (origin, duplicate) pair is violating the ITSM constrain.
        log.warning('origin %s has code greater than the duplicate %s'%
                        (origin_ticket, duplicate_ticket))
        duplicates_violation_code_number['origins'].append(origin_ticket)

        # try-except block is needed because origin_ticket might not be present
        # inside the df dataframe if this is the case, will artificially add NaN
        try:
            # TODO: convert following instructions in df.loc[df[column1]==1, column2][0]
            open_time = df[df['incidentid'] == origin_ticket]['opentime'].values[0]
            resolved_time = df[df['incidentid'] == origin_ticket]['resolvedtime'].values[0]
            closed_by = df[df['incidentid'] == origin_ticket]['closedby'].values[0]
        except IndexError:
            TicketNotExistWarning(origin_ticket)
            open_time, resolved_time, closed_by = (np.nan, np.nan, np.nan)

        # store all the relevant information about the violating constrain
        log.warning('storing violation constrain information')
        duplicates_violation_code_number['origins: open date'].append(open_time)
        duplicates_violation_code_number['origins: close date'].append(resolved_time)
        duplicates_violation_code_number['origins: closed by'].append(closed_by)

        duplicates_violation_code_number['duplicates'].append(duplicate_ticket)

        open_time = df[df['incidentid'] == duplicate_ticket]['opentime'].values[0]
        resolved_time = df[df['incidentid'] == duplicate_ticket]['resolvedtime'].values[0]
        closed_by = df[df['incidentid'] == duplicate_ticket]['closedby'].values[0]
        
        duplicates_violation_code_number['duplicates: open date'].append(open_time)
        duplicates_violation_code_number['duplicates: close date'].append(resolved_time)
        duplicates_violation_code_number['duplicates: closed by'].append(closed_by)

def check_reverse_order(df:pd.DataFrame, duplicates_dict:Dict[str, str]) -> None:
    '''
        This method is a follow-up check in case check_violation_duplicate_less_origin
        return any violating pairs (origin, duplicate). This check is necessary 
        because agents receive IM tickets acoording a heap order (newest at the top,
        and oldest at the bottom), so they might link IMs in a revers order. It meas
        duplicate <-- origin and not as should be i.e. origin <-- duplicate.

        A reverse link is not strictly correct but it might be acceptable.

        Parameters
        ----------
        df: pd.DataFrame
            Table of all tickets

        duplicates_dict: dict
            Dictionary of: ticket -> origin ticket (or star)

        Returns
        -------
        None
    '''
    
    for origin, star in duplicates_dict.items():
        if star is nx.DiGraph:
            tickets = star.nodes # list of all the tickets in star
            # switch origin and duplicate order
            tickets[0], tickets[1] = tickets[1], tickets[0]
            first_in_time_order = df[df['incidentid'] == tickets[0]]['opentime']
            for ticket in tickets:
                second_in_time_order = df[df['incidentid'] == ticket]['opentime']
                if first_in_time_order < second_in_time_order:
                    # Agent link reversely origin and duplicate
                    log.warning('star violating reverse order: %s'%origin)
                first_in_time_order = second_in_time_order

def check_violation_origin_type(duplicates_dict:Dict[str, str]) -> dict:
    '''
        This method check if the origin ticket is not an Incident.
        If this is the case, the origin ticket is violating an ITSM constrain:
        origin and duplicates have to be of the same ticket type.

        The execution is not interrupted, nevertheless the star is cleaned by
        not acceptable duplicate tickets.

        Parameters
        ----------
        duplicates_dict: dict
            Dictionary of: ticket -> origin ticket (or star)

        Returns
        -------
        duplicates_violation_origin_type: dict
            Dictionary of: origin ticket -> star of the origin ticket
    '''

    duplicates_violation_origin_type = {}
    for origin, duplicates in duplicates_dict.items(): 
        if type(duplicates) is not str and 'IM' not in origin:
            log.warning('duplicates is not of the correct ticket type')
            star = list(duplicates.nodes)
            star.remove(origin)
            duplicates_violation_origin_type[origin] = star
    return duplicates_violation_origin_type

def relevant_suspicious_users(suspicious_users:pd.DataFrame, threshold:int) -> list:
    '''
        This is a support method for check_violation_duplicate_less_origin_deeper_analysis.
        It is used to retrieve the subset of suspicious user violating the duplicate_less_origin
        constrain, more then threshold times.

        Parameters
        ----------
        suspicious_users: pd.DataFrame
            all the users violating the constrain

        threshold:int
            minimum number of violates that a user have to do to be considered 
            as a relevant suspicious user

        Returns
        -------
        list ofo relevant suspicious users.
    '''
    return list(suspicious_users.where(suspicious_users >= threshold)
            .dropna().sort_values(ascending=False).index)

def check_violation_duplicate_less_origin_deeper_analysis(duplicates:tuple, 
    violation_threshold:int = 10) -> tuple:
    '''
        This method is an extension of check_violation_duplicate_less_origin method:
        tickets violating the "duplicate_less_origin" constrain are clusterized according
        the operator.

        Parameters
        ----------
        duplicates: tuple
            duplicate tickets code violating the "less than origin" constrain

        violation_threshold:int
            Minimum nuymber of tickets violating the constrain, per operator.

        Returns
        -------
        None
    '''

    # DISC connection
    disc_con = pyodbc.connect('DSN=DISC DP Hive 64bit', autocommit=True)

    query = """
    SELECT recordid, modifytime, operator, type, description 
    FROM lab_dlb_eucso_uni_sla.itsm_ecb_disc_activity_clean 
    WHERE operator <> 'SM-AutoClose' and description regexp '.*[dD]uplicat.*' and 
    description regexp '((IM|RF)[0-9]+)' and recordid in {}
    """.format(duplicates)

    tickets_activities = pd.read_sql(sql=query, con=disc_con)
    # list of rows from tickets_activities with just 'recordid' and 'operator' columns
    filtered_tickets_activities = tickets_activities[['recordid', 'operator']].to_numpy().tolist()
    # consider unique rows: this is done because a single operator repeat the same action 
    #       (ex. duplicate of IM99999) for the same IM
    # in this way we have the unique list of operator breaking the constrain for each IM
    filtered_tickets_activities = [list(row) for row in set(tuple(row) 
            for row in filtered_tickets_activities)]
    filtered_tickets_activities = pd.DataFrame(filtered_tickets_activities, 
            columns=['recordid', 'operator']) # convert in Dataframe...
    # frequency of IM tickets  per each operator violating the constrain
    frequency_suspicious_operators = filtered_tickets_activities['operator'].\
            value_counts().to_frame('N. tickets')

    # count number of time each operator violates the constrain
    users_violating_code_number = filtered_tickets_activities['operator'].value_counts()
    # list of unique operators violating the constrain more than violation_threshold parameter
    users_violating_code_number = relevant_suspicious_users(users_violating_code_number, violation_threshold)
    # pick IM where the operator execute an "action" (note, we are dealing with
    #       action where the description is referring to a "duplication")
    incidents_of_suspicious_users = tickets_activities.loc[
            tickets_activities['operator'].isin(users_violating_code_number)]
    return incidents_of_suspicious_users, frequency_suspicious_operators
