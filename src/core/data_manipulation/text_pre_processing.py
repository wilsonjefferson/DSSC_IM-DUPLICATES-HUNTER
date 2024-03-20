import pandas as pd
import numpy as np
import re
import unicodedata
from tqdm import tqdm
from typing import List, Dict, Callable

import logging
log = logging.getLogger('textProcessingLogger')

from ConfigNameSpace import package_root, MAIN_STAGE
backup_location = MAIN_STAGE.backup_location
n_chunks = MAIN_STAGE.n_chunks

from core.data_manipulation.data_filtering import remove_no_valid_tickets
from core.utils.recovery_system import recovery_decorator

from nltk import SnowballStemmer
stemmer = SnowballStemmer('english')

import spacy
spacy.prefer_gpu()

spacy_dir = package_root+'spacy-data/en_core_web_lg-3.4.0/en_core_web_lg/en_core_web_lg-3.4.0'
nlp = spacy.load(spacy_dir)
nlp.Defaults.stop_words |= {'null'}


@recovery_decorator(['step_text_preprocessing/df_intermidiate_cleaned.xlsx', 
                    'step_text_preprocessing/list_intermidiate_no_valid_incidents_id.pickle'])
def text_processing(df:pd.DataFrame, method:Callable[[str], str], columns:List[str]) -> tuple:
    '''
        Support function of text_preprocessing.

        Perform text pre-processing according the passed method. 
        The pre-processing will remove stopwords, useless sentences and 
        normalize the text. After the filtering, no (more) avalid tickets 
        are deleted.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with the tickets

        method: object (custom function)
            Text filtering method

        Returns
        -------
        df: pd.DataFrame
            Filtered tickets after text pre-processing

        df_no_valid: pd.DataFrame
            No valid tickets (just for testing)
    '''

    clean = '_cleaned'
    new_columns = []
    for column in columns:
        new_columns.append(column + clean)
        df[column + clean] = text_filtering(method, df[column], column)

    # remove pre-processed empty strings cases
    df, df_no_valid = remove_no_valid_tickets(df, new_columns)
    return df, df_no_valid['incidentid'].tolist()

tqdm.pandas() # https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
def text_filtering(method:Callable[[str], str], df_series:pd.DataFrame, column:str) -> pd.DataFrame:
    '''
        Support function of text_processing.

        Effectively apply text filtering in chuncks: each chunck is preprocessed
        and stored in a file in case of unexpected interruption of the program.
        In case of interruption the program will not start from the first chunk 
        but it will start from the last before the interruption.

        This support function return the merge of all chunks in a dataframe.


        Parameters
        ----------
        method: object (custom function)
            Text filtering method

        df_series: pd.Series
            Pandas series with the text of the ticket

        column: str
            column name between 'Title' and 'Description'

        Returns
        -------
        results: pd.DataFrame
            merge of preprocessed text chunks
    '''
    
    # check if exist intermidiate backups
    from pathlib import Path
    import platform

    pltf = platform.system()

    if pltf == 'Windows':
        backup = backup_location+'step_text_preprocessing/text_filtering_chuncks\\'+column+'\\'
    else:
        backup = backup_location+'step_text_preprocessing/text_filtering_chuncks/'+column+'/'
    
    chunk_files = Path(backup).glob('**/*')
    chunk_files = [f for f in chunk_files if f.is_file()]

    # define the starting chunk number for the text processing
    start = 0 if len(chunk_files) == 0 else len(chunk_files)
    # split text into chunks
    list_of_df_series = np.array_split(df_series, n_chunks)
    list_of_df_series = list_of_df_series[start:]

    for idx, df in enumerate(tqdm(list_of_df_series, desc=f'Applying text filtering on df {column} chuncks')):
        idx += start
        log.info(f'chunk number: {idx}')
        df_tmp = df.progress_apply(lambda x: method(text_filter_no_content(str(x))))
        # store chunk in pickle file
        tmp_backup = Path(backup+'chunk_'+str(idx)+'.pickle')
        df_tmp.to_pickle(tmp_backup)

    log.info('text filtering completed')

    chunk_files = Path(backup).glob('**/*')
    chunk_files = [f for f in chunk_files if f.is_file()]

    results = pd.concat([pd.read_pickle(file) for file in chunk_files], axis=0)
    return results

def text_filter_no_content(request:str) -> str:
    '''
      The purpose of this method is to run the text filtering, focusing on the
      text content.

      Parameters
      ----------
      request: str
          String to be pre-processed

      Returns
      -------
      request_string: str
          The text pre-processed string
    '''
    log.debug('Ticket text: %s'%request)
    log.debug('Compile email foots and headers')
    foot1 = re.compile('[b|B]est\s[r|R]egards.*')
    foot2 = re.compile('[r|R]egards.*')
    foot3 = re.compile('[b|B][s|S],?\n.*')
    foot4 = re.compile('[c|C]heers?.*')
    header1 = re.compile('^.*=*([i|I]ncident=[r|R]eported|INCIDENT=REPORTED)=*')
    header2 = re.compile('^.*=([e|E]mail|EMAIL)=*')
    header3 = re.compile('^.*=*\s*([d|D]escription|DESCRIPTION)\s*[=|:]*')
    header4 = re.compile('^.*[d|D]ear[s|S]?')

    log.debug('Clean email foots')
    request_string = re.sub(foot1, '', request)
    request_string = re.sub(foot2, '', request_string)
    request_string = re.sub(foot3, '', request_string)
    request_string = re.sub(foot4, '', request_string)

    log.debug('Clean email headers')
    request_string = re.sub(header1, '', request_string)
    request_string = re.sub(header2, '', request_string)
    request_string = re.sub(header3, '', request_string)
    request_string = re.sub(header4, '', request_string)

    log.debug('Ticket text cleaned: %s'%request_string)
    return request_string

def soft_preprocess(request:str) -> str:
    '''
      The purpose of this method is to run the Soft text preprocessing.

      Parameters
      ----------
      request: str
          String to be pre-processed

      Returns
      -------
      request_string: str
          The text pre-processed string
    '''
    
    log.debug('Compile general characters')
    puncts = re.compile(r'[\'".:,;!?\\_\-‘’/<>]+') # punctuations
    parens = re.compile('[()[\]{}]') # parenthesis
    maths = re.compile('[=+*%]+') # matemathics
    single_char = re.compile('^[a-zA-Z]?$')
    multiple_spaces = re.compile('\s+')
    remove_space_init_end = re.compile('^(\s)+$')
    http_pattern = re.compile(    
        r'https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # or IP address
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # or IPv6 address
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$',  # path or query string
        re.IGNORECASE)
    date_pattern = re.compile(
        r'\b(?:\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[- ](?:\d{1,2}(?:st|nd|rd|th)?,?[- ]?)?\d{2,4})\b')
    date_time_pattern = re.compile(    
        r'(\d{4}-\d{2}-\d{2})'  # YYYY-MM-DD
        r'(?:T|\s)'  # Date-time separator (T or space)
        r'(\d{2}:\d{2}:\d{2})'  # HH:MM:SS
        r'(?:\.\d{1,6})?'  # Optional fractional seconds
        r'(?:Z|[+-]\d{2}:\d{2})?')  # Optional timezone (Z or +/-HH:MM))
    
    log.debug('Ticket text: %s'%request)
    
    log.debug('String in lower case')
    request = request.lower()
    log.debug('Expand contractions')
    request = expand_contractions(request)
    log.debug('Expand ECB Acronyms')
    request = expand_abbreviation(request)

    log.debug('Clean text from general characters')
    log.debug('date-time replaced with special token')
    request_string = re.sub(date_time_pattern, '[DATETIME]', request)
    log.debug('date replaced with special token')
    request_string = re.sub(date_pattern, '[DATE]', request_string)
    log.debug('http replaced with special token')
    request_string = re.sub(http_pattern, '[HTTP]', request_string)
    log.debug('urls removed')
    request_string = re.sub(puncts, ' ', request)
    log.debug('punctuation removed')
    request_string = re.sub(maths, ' ', request_string)
    log.debug('math chars removed')
    request_string = re.sub(parens, ' ', request_string)
    log.debug('parenthesis removed')
    request_string = re.sub(single_char, ' ', request_string)
    log.debug('single char removed')
    request_string = re.sub(multiple_spaces, ' ', request_string)
    log.debug('multispace removed')
    request_string = re.sub(remove_space_init_end, '', request_string)


    # substitute people names with special token [PERSON]
    nlp_text = nlp(request_string)
    request_string = ' '.join(['[PERSON]' if ent.label_ == 'PERSON' else token.text \
                                    for token, ent in zip(nlp_text, nlp_text.ents)])

    log.debug('Ticket text cleaned: %s'%request_string)
    return request_string

def expand_abbreviation(text:str) -> str:
    words = text.split()
    expanded_words = [dict_ecb_acronyms.get(word, word) for word in words]
    return ' '.join(expanded_words)

### read ECB Acronyms from excel file ###
import glob
root_sources = 'sources/text_preprocessing/'
folder_path = root_sources
xls_files = glob.glob(f"{folder_path}*.xls")

dfs = [pd.read_excel(file_path) for file_path in xls_files]
df_ecb_acronyms = pd.concat(dfs, axis=0, ignore_index=True)
dict_ecb_acronyms = df_ecb_acronyms.set_index('Acronym_lower')['Stands_for'].to_dict()
######################################################

def expand_contractions(text:str) -> str:
    for contraction, expanded_form in dict_contraction_mapping.items():
        text = text.replace(contraction, expanded_form)
    return text

### Read each Excel file into a separate DataFrame ###
df_contraction_mapping = pd.read_excel(root_sources+'contractions.xlsx', engine='openpyxl')
dict_contraction_mapping = df_contraction_mapping.set_index('contraction')['expansion'].to_dict()
######################################################
