import pandas as pd
from typing import List

from core.utils.Ticket import Ticket

def build_corpus(df:pd.DataFrame) -> tuple:
    '''
        This method receive all the documents and create a corpus list of sentences 
        and titles (since they might be used during the similarity measure).

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame of the dataset to use to train the model(s)

        Returns
        -------
            tuple
            Tuple of dataframe df and list corpus
    '''
    
    import pickle
    import os

    from ConfigNameSpace import MAIN_STAGE
    backup_location = MAIN_STAGE.backup_location
    location = backup_location + 'step_compute_similarities/'

    if os.path.exists(location+'list_corpus.pickle'):
        with open(location+'list_corpus.pickle', "rb") as f:
            corpus = pickle.load(f)
        df = pd.read_pickle(location+'df_tickets.pickle')
    else:
        from pandarallel import pandarallel
        pandarallel.initialize()
        
        df[['origin_ticket_cleaned', 'duplicate_ticket_cleaned']] = \
            df[['origin_text_cleaned', 'duplicate_text_cleaned']].parallel_applymap(lambda x: Ticket(str(x)))

        corpus = df['origin_ticket_cleaned'].tolist() + df['duplicate_ticket_cleaned'].tolist()

        df.to_pickle(location+'df_tickets.pickle')
        with open(location+'list_corpus.pickle', 'wb') as f:
            pickle.dump(corpus, f)

    return df, corpus

def doc_freq(corpus:list) -> dict:
    '''
        This method counts the word frequency among the documents: count the
        number of occurance of a word on a set of documents (but not the repetitions
        of the same word in the same document).

        Parameters
        ----------
        corpus: list
            List of Tickets (sentences and titles)

        Returns
        -------
        df_w_counter: dict
            Dictionary with word frequencies
    '''

    return _freq(corpus, True)
    
def word_freq(sentence:List[str]) -> dict:
    '''
        This method counts the word frequency in a sentence.

        Parameters
        ----------
        sentence: list
            List of tokens

        Returns
        -------
        df_w_counter: dict
            Dictionary with word frequencies
    '''

    return _freq([sentence], False)

def _freq(list_items:list, unique:bool) -> dict:
    '''
        This method counts the word frequency in a given collection.

        Parameters
        ----------
        list_items: list
            List of items (sentences or tokens)

        unique: bool
            True if doc_freq, False if word_freq

        Returns
        -------
        df_w_counter: dict
            Dictionary with word frequencies
    '''

    if unique:
        list_items = (set(list_item) for list_item in list_items)
    
    df_w_counter = dict()
    for list_item in list_items:
        for item in list_item:
            df_w_counter[item] = df_w_counter.get(item, 0) + 1
    return df_w_counter
    