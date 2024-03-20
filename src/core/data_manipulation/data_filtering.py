import pandas as pd
import pickle
import logging
from typing import List, Dict

from ConfigNameSpace import MAIN_STAGE


log = logging.getLogger('dataFilteringLogger')
backup_location = MAIN_STAGE.backup_location


def is_valid_type(obj) -> bool:
    if isinstance(obj, tuple):
        result = True
        for item in obj:
            result = result & all(ticket_type not in item for ticket_type in ['RF', 'SD'])
    else:
        result = all(ticket_type not in obj for ticket_type in ['RF', 'SD'])
    return result

def _isin(ids:list, row:list) -> bool:
    row = [[i] if type(i) == str else list(i) for i in row]
    row = [item for sublist in row for item in sublist]

    result = True    
    for item in row:
        result = result & (item in ids)
    return result

def is_empty(item:str) -> bool:
    '''
        This is a support method, it is used to verify if the tickets have the
        description or if it is empty. In this last case, False is returned,
        otehrwise True.

        Parameters
        ----------
        str_item: str
            String representing a ticket message

        Returns
        -------
            bool
            False if description is empty (or null or nan), True otherwise
    '''

    try:
        return item is None or item in ['nan', '', ' ']
    except TypeError as e:
        log.warning('%s: type:%s - %s'%(e, type(item), item))
    
def remove_no_valid_tickets(df:pd.DataFrame, columns:List[str]) -> pd.DataFrame:
    '''
        Remove tickets from input dataframe not respecting the following conditions:
        - ticket ID is not empty and it is of a specific type
        - passed columns are not empty or contains None

        Parameters
        ----------
        df: pd.DataFrame
            Tickets dataframe

        columns: list
            List of columns for df to focus on and to check the conditions

        Returns
        -------
        df: pd.DataFrame (tuple)
            df with tickets satisfying the conditions and df with tickets violating them
    '''

    id_empty = df.incidentid.apply(is_empty).to_numpy()
    id_no_type = ~df.incidentid.apply(is_valid_type).to_numpy()
    id_no_valid = id_empty | id_no_type

    empty = df[columns].applymap(is_empty).all(axis=1).to_numpy()
    none = df[columns].applymap(lambda x: x == None).all(axis=1).to_numpy()
    no_valid_rows = id_no_valid | empty | none

    return df[~no_valid_rows], df[no_valid_rows]

def define_sentences(df:pd.DataFrame, columns:List[str]) -> pd.DataFrame:
    '''
        This method is used to decide which text, between description and title,
        will be used as main text ticket. In general, description is prefered since
        it should bring more information but, if it is empty, the title is selected.

        Parameters
        ----------
        df: pd.DataFrame
            Tickets dataframe

        columns: list
            List of columns for df to focus on and to pick the representative ticket text

        Returns
        -------
            tuple
            df with text tickets and list of no valid ticket id
    '''

    df = df.dropna(subset=columns) # Sanity check

    # split each cell-text in list of words
    text_splitted = df[columns].applymap(str.split)
    print('text_splitted shape: %s %s\n'%text_splitted.shape)
    print(f'text_splitted:\n{text_splitted}\n')

    # True: cell-text with more than 1 word, False: otherwise
    text_not_unique = text_splitted.applymap(lambda x: len(x)>5).any(axis=1)
    print('text_not_unique shape: %s\n'%text_not_unique.shape)
    print(f'text_not_unique:\n{text_not_unique}\n')

    dff = df[text_not_unique]
    no_valid_incidentid = df[~text_not_unique].incidentid.tolist()

    return dff, no_valid_incidentid

def separate_stars_to_tickets(df:pd.DataFrame, dict_duplicates:Dict[str, str]) -> tuple:
    '''
        This method is used to create twho separated dictionaries, one just for
        for (duplicate_ticket_id, origin_ticket_id) and one just for 
        (duplicate_ticket_id, Star_Graph_Object).

        Parameters
        ----------
        df: pd.DataFrame
            Tickets dataframe

        dict_duplicates: dict
            dictionary with all recognized duplicate tickets, with also the stars

        Returns
        -------
            tuple
            dict_stars_filtered dictionary with just (duplicate_ticket_id, origin_ticket_id)
            dict_tickets_filtered dictionary with just (duplicate_ticket_id, Star_Graph_Object)
    '''
    import os
    import numpy as np

    # convert dictionary to pandas dataframe
    duplicates = pd.DataFrame(dict_duplicates.items(), columns=['duplicate','origin'])
    log.debug(duplicates.head())

    # keep just those rows where elements are string
    are_string = duplicates.applymap(lambda x: isinstance(x,  str)).all(axis = 1).to_numpy()
    log.debug('are_string - n.Tot: %s n. True: %s n. False: %s'%(are_string.size, 
                                                                 are_string.sum(), 
                                                                 are_string.size - are_string.sum()))

    # keep just those valid incident and present in df
    are_valid_file = backup_location+'step_construct_duplicates_dict/are_valid.npy'
    if os.path.exists(are_valid_file):
        with open(are_valid_file, 'rb') as f:
            are_valid = np.load(f, allow_pickle=True)
    else:
        check_validity = duplicates.applymap(is_valid_type).all(axis = 1)
        log.debug(f'validity:\n{check_validity}')

        from tqdm import tqdm
        tqdm.pandas()
    
        ids = list(df.incidentid)
        check_existency = duplicates.progress_apply(lambda x: _isin(ids, x), axis=1)

        are_valid = (check_validity & check_existency).to_numpy()
        log.debug(f'valid pairs:\n{are_valid}')

        with open(are_valid_file, 'wb') as f:
            np.save(f, are_valid, allow_pickle=True)
    
    log.debug('are_valid - n.Tot: %s n. True: %s n. False: %s'%(are_valid.size, 
                                                                are_valid.sum(), 
                                                                are_valid.size - are_valid.sum()))

    dict_stars_filtered = duplicates[are_string & are_valid]
    dict_stars_filtered = dict(zip(dict_stars_filtered.duplicate, dict_stars_filtered.origin))
    log.debug('dict stars filtered: %s'%len(dict_stars_filtered))

    dict_tickets_filtered = duplicates[(~are_string) & are_valid]
    dict_tickets_filtered = dict(zip(dict_tickets_filtered.duplicate, dict_tickets_filtered.origin))
    log.debug('dict tickets filtered: %s'%len(dict_tickets_filtered))

    with open(backup_location+'step_construct_duplicates_dict/dict_stars.pickle', 'wb') as handle:
        pickle.dump(dict_tickets_filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dict_stars_filtered, dict_tickets_filtered
