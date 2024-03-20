"""
    This script is the main of the project.
    From here the tickets are analyzed and grouped
    according different criterias.

    @file: main.py
    @version: v. 1.2.1
    @last update: 26/06/2022
    @author: Pietro Morichetti
"""

import re
import pandas as pd
import networkx as nx
from collections import Counter

from core.data_manipulation.data_handlers.SourceHandler import SourceHandler
from core.data_manipulation.construct_dataset import build_dataset
from core.data_manipulation.itsm_tickets_constrains import (
    check_violation_self_pointing_ticket, check_violation_duplicate_less_origin, 
    check_violation_origin_type, check_violation_duplicate_less_origin_deeper_analysis, 
    check_reverse_order)

from core.utils.CompareCodes import CompareCodes
from core.utils.plots.stars_plots import (stars_per_depth, stars_per_category, 
    plot_representative_stars_per_category)


def ticket_text_normalize(origin_tickets):
    for idx, ticket in enumerate(origin_tickets):
        ticket_type, ticket_digit = re.findall('\d+|\D+', ticket)
        origin_tickets[idx] = ticket_type.upper() + ticket_digit
    return origin_tickets

def get_origin_ticket(tickets:list):
    '''
        This method identify the origin ticket from
        a set of candidates.
        The origin ticket is the one with the lower
        digit part of the ticket code.

        Parameters
        ----------
        tickets: list
            List of candidates to the origin role

        Returns
        -------
        str
            The origin ticket
    '''

    splitted_tickets = {code: re.findall('\d+|\D+', code) for code in tickets}
    tickets_per_type = {}

    for tickets in splitted_tickets.values():
        ticket_type = tickets[0] # ticket type part
        if ticket_type not in tickets_per_type.keys():
            tickets_per_type[ticket_type] = []
        tickets_per_type[ticket_type].append(tickets[1]) # ticket digit part

    origin = None
    if 'IM' in tickets_per_type.keys():
        origin = 'IM' + min(tickets_per_type['IM'])
    elif 'SD' in tickets_per_type.keys():
        origin = 'SD' + min(tickets_per_type['SD'])
    elif 'RF' in tickets_per_type.keys():
        origin = 'RF' + min(tickets_per_type['RF'])

    if origin is None:
        print("set of origins: ", *tickets)

    return origin

def get_star_from_origin(duplicates_dict:dict, origin_ticket:str):
    '''
        Given a ""potential" origin, this method return the associated star.

        All origins are "real" origins until they are not pointing to
        another ticket (in that case we discover they are actually duplicate
        ticket).

        Parameters
        ----------
        duplicates_dict: dict
             Dictionary of: ticket -> origin ticket (or star)

        origin_ticket: str
            First ticket raised, first ticket of a series of tickets
            referring to the same "topic" 

        Returns
        -------
        nx.DiGraph and str
            The star and the origin ticket
    '''

    if not duplicates_dict.get(origin_ticket): # in case origin_ticket is faced for the first time and it is not present in duplicates_dict
        G = nx.DiGraph()
        duplicates_dict[origin_ticket] = G    
    elif type(duplicates_dict[origin_ticket]) is nx.DiGraph: # in case origin_ticket is a "real" origin
        G = duplicates_dict[origin_ticket]
    elif type(duplicates_dict[origin_ticket]) is str: # in case origin_ticket is actually a duplicate of another ticket
        origin_ticket = duplicates_dict[origin_ticket] # actual origin
        G = duplicates_dict[origin_ticket]
    return G, origin_ticket

def add_duplicates_in_star_origin(G:nx.DiGraph, origin_ticket:str, chain_origin_ticket:list, H:nx.DiGraph, duplicate_ticket:str):
    '''
        This method include all the ticket that are part of a "non-real" origin ticket
        to the current "real" origin ticket.

        Parameters
        ----------
        G: nx.DiGraph
             Star of the origin ticket

        origin_ticket: str
            First ticket raised, first ticket of a series of tickets
            referring to the same "topic" 

        duplicate_ticket: str
            A  non-first ticket, since was already reaised a previous
            ticket on the same "topic"

        H: nx.DiGraph
            Star of the duplicate ticket

        Returns
        -------
        None
    '''

    # INFO: nx.DiGraph is a mutable type, so it can be see as a reference obj passed to a function
    if H is nx.DiGraph: # in case we have discovered duplicate as no "real" origin, we move it's nodes (and edges) to the real origin
        G.add_edges_from(H.edges)
    else:
        G.add_node(H) # in case dupplicate is part of a graph
    
    if duplicate_ticket != origin_ticket: # avoid case of cyclic paths
        G.add_edge(duplicate_ticket, chain_origin_ticket) # create connection in the graph

def classify_ticket(origin_tickets:list, duplicate_words:list, rows_case_1_2_4:list, rows_case_3:list, rows_case_5:list):
    '''
        Classify tickets in three classes
        - rows_case_1_2_4: tickets in which there is one single ticket code in the text
            case_1 --> w1 w2 D1 w3 w4 C1 w5
            case_2 --> w1 w2 D1 C1 w3 C1 w4 w5
            case_4 --> w1 D1 w2 D2 w3 w4 C1 w5
        - rows_case_3: tickets in which there is one duplicate term and multiple ticket code in the text
            case_3 --> w1 w2 D1 w3 w4 D2 C1 w5
        - rows_case_5: tickets in which there are multiple duplicate term and multiple ticket code in the text
            case_5 --> w1 D1 w2 D2 w3 w4 C1 w5 C2

        Parameters
        ----------
        origin_tickets: str
            List of ticket codes in the text 

        duplicate_words: list
            List of duplicate terms

        rows_case_1_2_4: list
            List of tickets under one among case_1, case_2, case_4

        rows_case_3: list
            List of tickets under the case_3

        rows_case_5: list
            List of tickets under the case_5

        Returns
        -------
        None
    '''

    if len(origin_tickets) == 1: # find a single code in tickets
        rows_case_1_2_4.append(idx)
    elif len(duplicate_words) == 1 and len(origin_tickets) > 1: # find multiple codes and just 'duplicat*' word one in tickets
        rows_case_3.append(idx)
    elif len(duplicate_words) > 1 and len(origin_tickets) > 1: # find multiple 'duplicat*' and multiple codes in tickets
        rows_case_5.append(idx)


if __name__ == '__main__':

    regex1 = '.*duplicat.*'
    regex2 = '((IM|SD|RF)\d+)'

    source_handler = SourceHandler()
    df = source_handler.get_source() # \\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\Incidents 2019-2022.xlsx
    source_handler.get_destination()

    writer = pd.ExcelWriter(source_handler.destination, engine='openpyxl')

    rows_case_1_2_4 = list()
    rows_case_3 = list()
    rows_case_5 = list()

    duplicates_dict = {}
    duplicates_violation_code_number = {'origins': [], 'origins: open date': [], 'origins: close date': [], 'origins: closed by': [], \
                                        'duplicates': [], 'duplicates: open date': [], 'duplicates: close date': [], 'duplicates: closed by': []}
    duplicates_violation_same_code = {'origins': [], 'duplicates': []}

    compare_codes = CompareCodes()

    for idx in df.index:
        duplicate_words = re.findall(regex1, str(df.at[idx, 'solution']), flags=re.IGNORECASE)
        if len(duplicate_words) != 0:
            duplicate_ticket = df.at[idx, 'incidentid']
            tickets_codes = re.findall(regex2, str(df.at[idx, 'solution']), flags=re.IGNORECASE)
            origin_tickets = [pair[0] for pair in tickets_codes]
            origin_tickets = ticket_text_normalize(origin_tickets) # ticket type in upper case
            origin_tickets = list(set(origin_tickets)) # we need a unique list of ticket codes appearing in the same text

            # check violation: potential self-chain ticket
            check_violation_self_pointing_ticket(origin_tickets, duplicate_ticket, duplicates_violation_same_code)

            if len(origin_tickets) != 0:

                origin_ticket = get_origin_ticket(origin_tickets)

                # check violation: duplicate code less than origin code
                check_violation_duplicate_less_origin(compare_codes, df, origin_ticket, duplicate_ticket, duplicates_violation_code_number)

                '''
                    For each ticket we have to check that the duplicate code is already present in 
                    duplicates_dict:
                    - if it is present, it means duplicate_ticket has a graph G as value
                    - otherwise, duplicate_ticket was never seen in past and we can add in the origin_ticket graph.
                      However, we need to check if origin_ticket has a graph G as value of another ticket code, in that
                      case it means origin_ticket is part of a graph of another ticket code and we need to access this
                      graph

                    In general, duplicate_ticket should has a value the code of the origin ticket, and an
                    origin ticket should has as value the code of the another ticket or a graph (where it is the origin).
                '''

                # H is the graph of the "origin" duplicate_ticket or duplicate_ticket code if it is not present in duplicates_dict
                H = duplicates_dict[duplicate_ticket] if duplicates_dict.get(duplicate_ticket) else duplicate_ticket
                chain_origin_ticket = origin_ticket # in case origin_ticket is not a "real" origin
                G, origin_ticket = get_star_from_origin(duplicates_dict, origin_ticket)

                if duplicates_dict.get(duplicate_ticket):
                    if type(duplicates_dict[duplicate_ticket]) is not nx.DiGraph:
                        duplicates_dict[duplicate_ticket] = origin_ticket
                else:
                    duplicates_dict[duplicate_ticket] = origin_ticket

                # include duplicate(s) node(s) in the origin star
                add_duplicates_in_star_origin(G, origin_ticket, chain_origin_ticket, H, duplicate_ticket)
                
            # after checked violations and stored duplicates, let's classify tickets in cases...
            classify_ticket(origin_tickets, duplicate_words, rows_case_1_2_4, rows_case_3, rows_case_5)

    # save in excel file
    df.iloc[rows_case_1_2_4, :].to_excel(writer, sheet_name='case 1 2 4')
    df.iloc[rows_case_3, :].to_excel(writer, sheet_name='case 3')
    df.iloc[rows_case_5, :].to_excel(writer, sheet_name='case 5')

    df_violation = pd.DataFrame(data=duplicates_violation_code_number)
    df_violation.to_excel(writer, sheet_name='violation no increment')

    incidents_of_suspicious_users, frequency_suspicious_operators = check_violation_duplicate_less_origin_deeper_analysis(tuple(duplicates_violation_code_number['duplicates']))
    incidents_of_suspicious_users.to_excel(writer, sheet_name='suspicious_opeators')
    frequency_suspicious_operators.to_excel(writer, sheet_name='suspicious_opeators_freq')


    df_violation = pd.DataFrame(data=duplicates_violation_same_code)
    df_violation.to_excel(writer, sheet_name='violation same code')

    # check violation: origin and duplicates cannot be of different ticket type
    duplicates_violation_origin_type = check_violation_origin_type(duplicates_dict)
    df_violation = pd.DataFrame.from_dict(duplicates_violation_origin_type, orient='index')
    df_violation = df_violation.transpose()
    df_violation.to_excel(writer, sheet_name='violation origin type')

    stars_depth_dict = stars_per_depth(duplicates_dict)
    stars_dict = {}
    for origin, n_duplicates in stars_depth_dict.items():
        key = 'N. Duplicates: {}'.format(n_duplicates)
        if not stars_dict.get(key):
            stars_dict[key] = []
        stars_dict[key].append(origin)

    res = Counter(stars_depth_dict.values())
    # print(res)

    stars_per_category_dict = stars_per_category(duplicates_dict, stars_depth_dict)
    plot_representative_stars_per_category(stars_per_category_dict)

    columns_sotr_stars_per_n_tickets = ['N. Duplicates: {}'.format(n_tickets) for n_tickets in res.keys()]
    df_stars = pd.DataFrame.from_dict(stars_dict, orient='index').transpose()
    df_stars = df_stars[columns_sotr_stars_per_n_tickets]
    df_stars.to_excel(writer, sheet_name='Stars Origins')

    writer.save()

    if 'IM541829' in df['incidentid']:
        print('IM541829 --> ', df[df['incidentid']=='IM541829']['description'].values[0])

    check_reverse_order(df, duplicates_dict)
    build_dataset(df, duplicates_dict, [r'\\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\dataset.xlsx'])

    # destination = r'\\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\Incidents.xlsx'
    # df.to_excel(destination)
