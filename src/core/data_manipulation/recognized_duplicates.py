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
import networkx as nx
import pandas as pd
import logging
from typing import List, Dict
import pickle

from ConfigNameSpace import MAIN_STAGE


log = logging.getLogger('recognizeDuplicatesLogger')
backup_location = MAIN_STAGE.backup_location
make_plots = MAIN_STAGE.make_plots


from core.data_manipulation.data_filtering import separate_stars_to_tickets


def ticket_text_normalize(origin_tickets:List[str]) -> list:
    '''
        This method normalize the ticket ID, because sometimes agents
        report ticket ID in uppercase or in lowercase.

        Parameters
        ----------
        origin_tickets: list
            List of candidates to the origin role

        Returns
        -------
        origin_tickets: list
            List of candidates to the origin role
    '''

    for idx, ticket in enumerate(origin_tickets):
        ticket_type, ticket_digit = re.findall('\d+|\D+', ticket)
        origin_tickets[idx] = ticket_type.upper() + ticket_digit
    return origin_tickets

def get_origin_ticket(tickets:List[str]) -> str:
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

    log.debug('Split tickets in code and number')
    splitted_tickets = {code: re.findall('\d+|\D+', code) for code in tickets}
    tickets_per_type = {}

    for tickets in splitted_tickets.values():
        ticket_type = tickets[0] # ticket type portion
        if ticket_type not in tickets_per_type.keys():
            log.debug('Ticket tipe %s was never seen before'%ticket_type)
            tickets_per_type[ticket_type] = []
        tickets_per_type[ticket_type].append(tickets[1]) # ticket digit portion
    log.debug('Ticket type: %s'%tickets_per_type.keys())

    log.debug('Find the oriign by checking the numbers')
    origin = None
    if 'IM' in tickets_per_type.keys():
        origin = 'IM' + min(tickets_per_type['IM'])
    elif 'SD' in tickets_per_type.keys():
        origin = 'SD' + min(tickets_per_type['SD'])
    elif 'RF' in tickets_per_type.keys():
        origin = 'RF' + min(tickets_per_type['RF'])

    if origin is None:
        log.critical("set of origins: %s"%tickets)

    return origin

def get_star_from_origin(duplicates_dict:Dict[str, str], origin_ticket:str) -> tuple:
    '''
        Given a "potential" origin, this method return the associated star.

        All origins are "real" origins until they are not pointing to
        another ticket (in that case we discover they are actually duplicate
        ticket).

        The scope of this function is to (eventually) adjust the pair (origin, star)
        in case an origin has as value another key (in this case the origin is a duplicate)
        amd to return the pair (origin, star)

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

    if not duplicates_dict.get(origin_ticket):
        # in case origin_ticket is faced for the first time 
        # and it is not present in duplicates_dict
        log.debug('%s never seen before'%origin_ticket)
        log.debug('Creating a star for %s'%origin_ticket)
        G = nx.DiGraph() # create an empty start
        duplicates_dict[origin_ticket] = G # assign the start to the origin
    elif type(duplicates_dict[origin_ticket]) is nx.DiGraph:
        # in case origin_ticket is a "real" origin, so origin has a star has value
        log.debug('%s is a known origin'%origin_ticket)
        G = duplicates_dict[origin_ticket]
    elif type(duplicates_dict[origin_ticket]) is str:
        # in case origin_ticket is actually a duplicate of another ticket
        # so it means origin_ticket has as value another ticket
        log.debug('%s is poiting to another origin, it is a duplicate of %s'%
                      (origin_ticket, duplicates_dict[origin_ticket]))
        origin_ticket = duplicates_dict[origin_ticket] # actual origin
        G = duplicates_dict[origin_ticket]
    return G, origin_ticket

def add_duplicates_in_star_origin(G:nx.DiGraph, origin_ticket:str, 
    chain_origin_ticket:List[str], 
    H:nx.DiGraph, duplicate_ticket:str) -> None:
    '''
        This method include all the ticket that are part of a "non-real" origin 
        ticket to the current "real" origin ticket.

        Note:
        "real" origin ticket is the current origin of a star, however it is a 
        potential origin until all the IM tickets are checked (it is possible 
        that the current origin is actually a duplicate of another 
        IM ticket not yet analyzed).
        "non-real" origin ticket is a ticket that is no longer the current origin
        of a star since is is discovered another IM ticket as origin.

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

    # INFO: nx.DiGraph is a mutable type, so it can be see as a 
    # reference obj passed to a function

    if H is nx.DiGraph:
        # in case we have discovered duplicate as no "real" origin, 
        # we move it's nodes (and edges) to the real origin
        log.debug('Star H is included in star G')
        G.add_edges_from(H.edges)
    else:
        log.debug('%s is a single ticket and will be included in G star'%H)
        G.add_node(H) # in case dupplicate is part of a graph
    
    if duplicate_ticket != origin_ticket: # avoid case of cyclic paths
        G.add_edge(duplicate_ticket, chain_origin_ticket) # graph connection

def search_recognized_duplicates(df:pd.DataFrame) -> tuple:
    '''
        This method looks in the solution of each ticket to understand if the actual
        ticket is actually a recognized duplicates by the agents.
        The method create a dictionary (key:duplicate -> value:origin) where:
        - duplicate is the ticket ID (string)
        - origin is a ticket ID or a Graph (string/nx.DiGraph)
        The Graph is a representation of the interconenction among duplicates and their
        origin, it can be seen as a "Star" where the core node is the really first ticket
        and all the other nodes are its duplicates, concatenated to the origin node.

        Keeping in mind the structure of Dictionary and the fact that multiple 
        duplicates may share the same origin, the duplicates are used as key
        and origins as values.

        The case (key:duplicate -> value:Graph) is encountered when the
        "duplicate" is actually a core node of a star (i.e. an origin) and the Graph
        is the corrisponding star. So, the "duplicate" is actually an origin of a star.
        
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with all the tickets
        
        Returns
        -------
        dict_stars_filtered: dict
            Dictionary with only key:duplicate (ticket ID) -> value:origin (ticket ID)
        
        dict_tickets_filtered: dict
            Dictionary with only key:duplicate (ticket ID) -> value: nx.DiGraph (star)
    '''

    regex1 = '.*duplicat.*'
    regex2 = '((IM|SD|RF)\d+)'
    duplicates_dict = {}

    if type(df) != pd.DataFrame and len(df) == 0:
        raise Exception('df is not a DataFrame!')            

    for idx in df.index:
        log.debug('%s - collect duplicate terms'%idx)
        duplicate_words = re.findall(regex1, str(df.at[idx, 'solution']), flags=re.IGNORECASE)
        if len(duplicate_words) != 0:
            duplicate_ticket = df.at[idx, 'incidentid']
            log.debug('%s - a duplicate ticket %s is found'%(idx, duplicate_ticket))
            tickets_codes = re.findall(regex2, str(df.at[idx, 'solution']), flags=re.IGNORECASE)
            origin_tickets = [pair[0] for pair in tickets_codes]
            origin_tickets = ticket_text_normalize(origin_tickets) # ticket type in upper case
            # we need a unique list of ticket codes appearing in the same text
            origin_tickets = list(set(origin_tickets))
            log.debug('%s - list of origins for %s: %s'%(idx, duplicate_ticket, origin_tickets))

            if len(origin_tickets) != 0:
                origin_ticket = get_origin_ticket(origin_tickets)
                log.debug('%s - %s choosen origin for %s'%(idx, origin_ticket, duplicate_ticket))

                '''
                    For each ticket we have to check that the duplicate code is 
                    already present in duplicates_dict:
                    - if it is present, it means duplicate_ticket has a graph G as value
                    - otherwise, duplicate_ticket was never seen in past and we 
                      can add in the origin_ticket graph.
                      However, we need to check if origin_ticket has a graph G 
                      as value of another ticket code, in that case it means 
                      origin_ticket is part of a graph of another ticket code 
                      and we need to access this graph

                    In general, duplicate_ticket should has a value the code of 
                    the origin ticket, and an origin ticket should has as value 
                    the code of the another ticket or a graph (where it is the origin).

                    Warning: It is important to notice that, even if the dataframe of incidents
                    is ordered from oldest to newest incident, the order does not guarantee to 
                    find duplicate tickets "easily" associated to an existing star. This because
                    it was observed that agents, dealing with incident tickets, are not always 
                    able to immediatelly spot that a certain ticket is duplicate of an already
                    existing "chain". Therefore, it may happen that agents implicitly build a star which,
                    at the end, it is discovered being part of another oldest star, this because they found out
                    that the origin of a star is actually duplicate of another node of an already existing star.
                '''
                # H is the graph of the "origin" duplicate_ticket or duplicate_ticket 
                # code if it is not present in duplicates_dict
                log.debug('%s - define origin and star'%idx)
                H = duplicates_dict[duplicate_ticket] if duplicates_dict.get(duplicate_ticket) else duplicate_ticket
                chain_origin_ticket = origin_ticket # in case origin_ticket is not a "real" origin
                G, origin_ticket = get_star_from_origin(duplicates_dict, origin_ticket)

                if duplicate_ticket == origin_ticket:
                    log.debug('%s - origin and duplicate are the same (%s), this is a loop!'%(idx, origin_ticket))
                    continue # stop here and move to the next idx in the loop
                
                log.debug('%s - storing duplicate/origin in duplicates dictionary'%idx)
                if duplicates_dict.get(duplicate_ticket):
                    if type(duplicates_dict[duplicate_ticket]) is not nx.DiGraph:
                        duplicates_dict[duplicate_ticket] = origin_ticket
                else:
                    duplicates_dict[duplicate_ticket] = origin_ticket

                # include duplicate(s) node(s) in the origin star
                log.debug('%s - add duplicates in star origin'%idx)
                add_duplicates_in_star_origin(G, origin_ticket, chain_origin_ticket, 
                    H, duplicate_ticket)

    log.debug('duplicates_dict: %s'%len(duplicates_dict))
    with open(backup_location+'step_construct_duplicates_dict/duplicates_dict.pickle', 'wb') as handle:
        pickle.dump(duplicates_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    log.debug('separate start to tickets')
    
    dict_stars_filtered, dict_tickets_filtered = separate_stars_to_tickets(df, 
                                                                        duplicates_dict)

    if make_plots:
        from core.utils.plots.stars_plots import (
            plot_scatter_stars,
            stars_per_depth,
            stars_per_category,
            plot_representative_stars_per_category,
            plot_duplicates_dates_frequency)

        plot_scatter_stars(dict_tickets_filtered)

        # plot the distribution of duplicates over the years
        duplicates_dates = [star.nodes for star in dict_tickets_filtered.values()]
        duplicates_dates = [df[df['incidentid'].isin(nodes)]['opentime'].tolist() for nodes in duplicates_dates]
        plot_duplicates_dates_frequency(duplicates_dates)

        stars_depth_dict = stars_per_depth(duplicates_dict)
        log.debug(f'stars_depth_dict: {len(stars_depth_dict)}')

        stars_per_category_dict = stars_per_category(duplicates_dict, stars_depth_dict)
        log.debug(f'stars_per_category_dict: {len(stars_per_category_dict)}')

        plot_representative_stars_per_category(stars_per_category_dict)

    return dict_stars_filtered, dict_tickets_filtered
