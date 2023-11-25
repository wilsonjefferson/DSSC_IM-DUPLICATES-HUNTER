import pandas as pd
import numpy as np
import logging
import networkx as nx
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import List, Dict, Callable

from ConfigNameSpace import DATE_FORMAT, TODAY, MAIN_STAGE, MAIN_BUILD, MAIN_DEPLOY

if MAIN_STAGE is MAIN_DEPLOY:
    log = logging.getLogger('modelDeployComponentsLogger')
else:
    log = logging.getLogger('modelBuildComponentsLogger')

backup_location = MAIN_STAGE.backup_location
build_catalog = MAIN_STAGE.build_catalog
make_plots = MAIN_STAGE.make_plots
destination = MAIN_STAGE.destination if MAIN_STAGE is MAIN_DEPLOY else None

import core
from core.data_manipulation.data_filtering import remove_no_valid_tickets, define_sentences
from core.data_manipulation.text_pre_processing import text_filter_no_content, soft_preprocess

from core.utils.recovery_system import recovery_decorator


from functools import wraps
def staging_decorator(*dec_args):
    '''
        This decorator is used to decide if activate or not the recovery system.
        The recovery system is used just during training and building the catalog,
        it is not used when effectively using the model in production since the program
        would require much less time to elaborate a prediction.
        In case of failure, the user just need to run again the program in DEPLOY mode
        and the lost prediction will be computed again.

        Parameters
        ----------
        dec_args: list
            List of argument for the recovery decorator

        Returns
        -------
        decorator: object
            Inner decorator
    '''
    logging.debug('dec_args: %s'% ', '.join(map(str, dec_args)))
    def decorator(func):
        '''
            Inner decorator.

            Parameters
            ----------
            func: object
                Function in deploy components to be executed

            Returns
            -------
            wrapper: object
                Inner decorator
        '''
        logging.debug('func: %s'%func)
        if build_catalog:
            logging.info('activate recovery system')
            func = recovery_decorator(*dec_args)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            '''
                Inner decorator.

                Parameters
                ----------
                args: list
                    List of argument for the deploy components function
                
                kwargs: dictionary
                    Dictionary of argument for the deploy components function

                Returns
                -------
                Return of function
            '''
            logging.debug('args: %s' % ', '.join(map(str, args)))
            return func(*args, **kwargs)
        return wrapper
    return decorator

tqdm.pandas() # https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
text_filtering = lambda method, df_series: df_series.progress_apply(lambda x: method(text_filter_no_content(str(x))))

def import_data(source:str = 'lab_dlb_eucso_uni_sla.itsm_incidents', query:str = None) -> tuple:
    '''
        Import data from DISC or an excel file.

        Parameters
        ----------
        None

        Returns
        -------
        df_original: pd.DataFrame
            Dataframe representation of the data
    '''
    from core.data_manipulation.data_handlers.SourceHandler import SourceHandler
    logging.info('\n##### IMPORT DATA #####\n')

    source_handler = SourceHandler()
    source_handler.connect('DISC', source)
    df_target = source_handler.get_data(query)

    logging.info(df_target.info())

    return df_target

@staging_decorator(['build_filter_no_valid_tickets.xlsx'])
def filter_invalid_tickets(df_target:pd.DataFrame) -> pd.DataFrame:
    '''
        Filter no valid tickets from original dataframe.

        Parameters
        ----------
        df_original: pd.DataFrame
            Dataframe with original data

        Returns
        -------
        df: pd.DataFrame
            Dataframe with just valid tickets
    '''
    
    df, _ = remove_no_valid_tickets(df_target, ['description', 'title'])
    df['text'] = df.title + pd.Series(len(df)*['. ']) + df.description
    
    log.debug(df)
    return df

@staging_decorator(['step_reduce_search_space/2_weeks_origins.xlsx', 
                'step_reduce_search_space/list_out_dated_origins.pickle'])
def reduce_search_space(df_target:pd.DataFrame) -> tuple:
    '''
        Remove outdated origins from the catalog (if exist).

        Parameters
        ----------
        df_target: pd.DataFrame
            Dataframe with incidents retrieved from DISC

        Returns
        -------
        df_target: pd.DataFrame
            Dataframe filtered from already seen incidents today

        df_2_weeks_origins: pd.DataFrame
            Dataframe of 2 weeks of origins from the catalog
        
        list_outdated_origins: list
            List of outdated origins that are less likely to be 
            the origin of newly raised incidents
    '''
    logging.info('\n##### REDUCE SEARCH SPACE #####\n')

    df_2_weeks_origins = None
    if not build_catalog:
        import pickle
        
        last_2_weeks = (datetime.today() - timedelta(days = 14)).strftime(DATE_FORMAT)
        build_backup_location =  MAIN_BUILD.backup_location
        print(f'build location: {build_backup_location}')
        df_2_weeks_origins = pd.read_excel(build_backup_location + 'catalog.xlsx', engine='openpyxl')
        df_2_weeks_origins, list_outdated_origins = time_search_space(df_2_weeks_origins, last_2_weeks, TODAY)
        logging.info('reduced space: %s\noutadated origins: %s'%
                     (df_2_weeks_origins.shape[0], len(list_outdated_origins)))
        
        with open(build_backup_location + 'catalog.pickle', 'rb') as handle: # r'D:\dict_stars.pickle'
            dict_stars = pickle.load(handle)

        for k in list_outdated_origins:
            dict_stars.pop(k, None)

        # exclude already seen incidents from today
        daily_origins = df_target[df_target.incidentid.isin(list(dict_stars.keys()))].incidentid.tolist()
        already_seen_incidents = [list(dict_stars[k].nodes) for k in dict_stars.keys() if k in daily_origins]
        already_seen_incidents = [incidentid for l in already_seen_incidents for incidentid in l]
        df_target = df_target[~df_target.incidentid.isin(already_seen_incidents)]
    else:
        first_origin = df_target.iat[0, 0]
        logging.info('First origin: %s'%first_origin)

        dict_stars = {}
        dict_stars[first_origin] = nx.DiGraph()
        dict_stars[first_origin].add_node(first_origin)

    return df_target, df_2_weeks_origins, dict_stars

def time_search_space(df:pd.DataFrame, start_time:pd.Timestamp, end_time:pd.Timestamp, format:str = DATE_FORMAT) -> tuple:
    '''
        This method filter tickets not in a given time-range.
        start_time < end_time.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame of tickets

        start_time: pd.Timestamp
            Oldest time to consider

        end_time: pd.Timestamp
            Early time to consider

        format: str
            Daytime format

        Returns
        -------
            tuple
            Filtered ticket dataframe and list of no valid ticket ID
    '''

    start_time = pd.to_datetime(start_time, format=format)
    end_time = pd.to_datetime(end_time, format=format)
    df['opentime'] = pd.to_datetime(df['opentime'], format=format)
    time_range = (df['opentime'] >= start_time)  & (df['opentime'] < end_time)
    return df[time_range], list(df[[not i for i in time_range]]['incidentid'])

@staging_decorator(['step_text_preprocessing/build_df_cleaned.xlsx'])
def text_preprocessing(df:pd.DataFrame) -> tuple:
    '''
        Compute text preprocessing and filter dict_stars_filtered from
        no valid pairs. Then, initialize DatasetBuilder.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with just valid tickets

        dict_stars_filtered: dict
            dictionary with duplicates and origins

        Returns
        -------
        dataset_builder: DatasetBuilder
            Object to build the dataset for the model(s)

        df: pd.DataFrame
            Filtered df after text pre-processing
    '''

    logging.info('\n##### COMPUTE TEXT PREPROCESSING #####\n')

    df, _ = text_processing(df, soft_preprocess, ['text'])
    df, _ = define_sentences(df, ['text_cleaned'])
    logging.debug(df.head())
    return df

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
        df[column + clean] = text_filtering(method, df[column])
    
    # remove pre-processed empty strings cases
    df, df_no_valid = remove_no_valid_tickets(df, new_columns)
    return df, df_no_valid['incidentid'].tolist()

@staging_decorator(['step_assess_on_new_tickets/dict_stars_evaluation.pickle', 
                    'step_assess_on_new_tickets/dict_recognized_duplicate.pickle', 
                    f'pred_probs/{TODAY}_prediction_proba_matrix.joblib'])
def assess_on_new_tickets(dict_model:dict, df:pd.DataFrame, dict_stars:Dict[str, str], df_2_weeks_origins:pd.DataFrame) -> tuple:
    '''
        Main sub workflow, for each new ticket:
        1. take a slice of the main dataframe (or all in case catalog exist)
        2. construct dataset for the new ticket
        3. compute similarities
        4. evaluate model on the computed similarities

        The model will infer if the new ticket is a duplicate or a new origin.


        Parameters
        ----------
        model: object
            Trained model to recognize duplicate incident tickets

        df: pd.DataFrame
            Main dataframe with newly raised tickets

        dict_stars: dict
            Dictionary of origin and stars

        start: integer
            Index from where to start the investigation
        
        df_tmp: pd.DataFrame
            Adjasted main dataframe

        Returns
        -------
        dict_stars: dict
            Dictionary of origin and stars

        dict_recognized_duplicates: dict
            Dictionary of new incident tickets and dataframe of its measures

        prediction_proba_matrix: np.matrix
            Numpy matrix where n_column is equal to each incident examinated and 
            n_rows is the current size of the catalog. So, each column is the array
            of prediction proba associated with a single potential duplicate IM ticket.
    '''

    """
        If the catalog need to be constructed than by the fault the first origin
        in the main dataframe is the very fisrst origin found (since no others exist).
        In this case we can set the start variable equals to 2 because, in order to
        build the catalog, we need to consider a "view" of the main dataframe 
        incrementally. In  this way, everytime we new ticket is one time a potential 
        duplicate and the other time a potential origin for the next ticket.

        The start variable is set to 2 because the count is from 0 to 1:
        main_dataframe.row(0) = origin
        main_datafrmae.row(1) = ticket under investigation (duplicate or new origin?)

        On the other hand, if the catalog already exist the start variable is 
        equals to 0, since we can investigate from the first ticket in the main 
        dataframe. 
        Moreover, the main dataframe is adjusted i.e. it is:
        main_dataframe = main_dataframe + catalog_dataframe
        this in order to later compare the new ticket text vs catalog texts.
    """

    if not build_catalog:
        df_tmp = pd.concat([df, df_2_weeks_origins])
        start = 0
        # setting df_tmp as concatenation between df and 2-weeks dataset is made on porpouse
        # in order, later on in the "construct dataset" part, to make use of the 
        # DatasetBuilder.build_data() function. This function needs to have all the
        # information inside the self.df dataframe which, in this case, is df_tmp.
    else:
        df_tmp = df
        start = 2

    dict_recognized_duplicates = {}

    if build_catalog:
        desc = 'Building catalog from given time-window of incidents'
    else:
        desc = 'Assessing incidents raised today %s'%TODAY
    
    prediction_proba_matrix = list()
    for idx in tqdm(range(start, len(df)), desc=desc, leave=False):
        # setting the upperboud for the range of this for loop as len(df)
        # ensure that, in production, to use and evaluate just the new incidets
        # In fact, setting len(df_tmp) means to check also the already identified
        # origins, that is wrong, since df_tmp = df + df_2_weeks_origins.

        new_ticket = df.iat[idx-1, 0] if build_catalog else df.iat[idx, 0]
        logging.info('Next new ticket: %s'%new_ticket)

        df_tmp_partial = df_tmp.iloc[:idx, :] if build_catalog else df_tmp
    
        # CONSTRUCT DATASET
        df_duplicates, df_dataset = construct_dataset(df_tmp_partial, dict_stars, 
                                                      new_ticket, len(df))

        # COMPUTE SENTENCES METRICS
        df_sms, metrics = compute_similarities(df_dataset)
    
        # MODEL EVALUATION
        dict_stars, df_duplicates, y_true_pred_proba = run_prediction(dict_model, new_ticket, dict_stars, 
                                                           df_duplicates, df_sms, metrics)
        
        prediction_proba_matrix.append(y_true_pred_proba)

    prediction_proba_matrix = np.column_stack(prediction_proba_matrix)
    
    logging.info('n. of origins: %s\nn. reconized duplicates: %s'%
                 (len(dict_stars), len(dict_recognized_duplicates)))
    return dict_stars, dict_recognized_duplicates, prediction_proba_matrix

def construct_dataset(df_tmp:pd.DataFrame, dict_stars:Dict[str, str], new_ticket:str, df_size:int) -> tuple:
    '''
        Construct the following datasets:
        - catalog origins vs new ticket
        - dataset to be used for the metrics

        Parameters
        ----------
        df_tmp: pd.DataFrame
            Main dataframe adjusted
        
        dict_stars: dict
            Dictionary of origin and stars
        
        new_ticket: str
            Newly raised incident ticket
        
        df_size: integer
            Size of the Main dataframe
        
        Returns
        -------
        df_duplicates: pd.DataFrame
            DataFrame of origins catalog and new ticket

        df_dataset: pd.DataFrame
            Dataframe to be evaluated by the trained model
    '''
    
    from core.data_manipulation.DatasetBuilder import DatasetBuilder
    logging.info('\n##### CONSTRUCT DUPLICATES DATASET #####\n')

    df_incidentid = list(dict_stars.keys())
    dict_duplicates = {}

    # TODO: check if df_size can be replaced with len(dict_stars)
    for k, v in list(zip(df_incidentid, [new_ticket]*df_size)):
        dict_duplicates[k] = v

    df_duplicates = {'duplicate':dict_duplicates.keys(), 'origin':dict_duplicates.values()}
    df_duplicates = pd.DataFrame.from_dict(df_duplicates)

    logging.info('\n##### CONSTRUCT DATASET #####\n')
    df_tmp = df_tmp.drop_duplicates(subset=['incidentid'])
    dataset_builder = DatasetBuilder(df_tmp, dict_duplicates, False)
    dataset_columns = dataset_builder.build_columns(True)
    df_dataset = dataset_builder.build_data(dict_duplicates, None, None)

    df_dataset = pd.DataFrame(df_dataset, columns=dataset_columns)
    return df_duplicates, df_dataset

def compute_similarities(df_dataset:pd.DataFrame) -> tuple:
    '''
        Given the dataset, calculate the sentence pair similarities.

        Parameters
        ----------
        df_dataset: pd.DataFrame
            Dataframe to train/test (and stress test) the model

        Returns
        -------
        metrics: list
            List of metrics name used to evaluate sentence similarity

        df_sms: pd.DataFrame
            Dataframe with just similarities for each pair of tickets
    '''
    
    from core.metrics_and_models.metrics import compute_metrics
    logging.info('\n##### COMPUTE SENTENCES METRICS #####\n')

    metrics, df_sms = compute_metrics(df_dataset,
        destination=backup_location+'build_data_measure.xlsx',
        custom_name_png=f'{TODAY}_')
    
    log.debug(metrics)
    log.debug(df_sms)
    return df_sms, metrics

def run_prediction(dict_model:dict, new_ticket:str, dict_stars:Dict[str, str], df_duplicates:pd.DataFrame, \
                   df_sms:pd.DataFrame, metrics:List[str]) -> tuple:
    '''
        Run the trained model to infer on the category of the newly raised incident
        ticket: 
        - If the ticket is not a duplicate than it is a new origin 
        - If the ticket is a duplicate than
            - if it has multiple origins detected by the model, the one with 
                mean metrics is considered and picked


        Parameters
        ----------
        dict_model: dict
            Trained model to recognize duplicate incident tickets

        new_ticket: str
            Newly raised incident ticket

        dict_stars: dict
            Dictionary of origin and stars
        
        df_duplicates: pd.DataFrame
            DataFrame of origins catalog and new ticket 
        
        df_sms: pd.DataFrame
            Dataframe with just similarities for each pair of tickets
        
        metrics: list
            List of metrics name used to evaluate sentence similarity

        Returns
        -------
        dict_stars: dict
            Updated dictionary of origin and stars

        df_duplicates: pd.DataFrame
            Updated dataFrame of origins catalog and new ticket

        y_true_pred_proba: np.array
            Numpy array of true prediction probabilities
    '''

    import numpy as np
    logging.info('\n##### MODEL EVALUATION #####\n')

    X = np.array(df_sms.loc[:, metrics])

    model = dict_model['clf']

    # use the best threshold to influence model prediction
    y_true_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    list_y_idx_true = np.where(y_pred == 1)[0]

    logging.debug('------- DUPLICATES -------\n %s'%df_duplicates)

    if len(list_y_idx_true) == 0:
        # no duplication, current ticket is an origin
        dict_stars = _create_star_origin(new_ticket, dict_stars)
        logging.info('%s is an origin'%new_ticket)
        return dict_stars, None, y_true_pred_proba
    else:
        origin, y_idx_true = _get_origin(df_duplicates, df_sms, list_y_idx_true, metrics)
        dict_stars = _store_in_star_origin(origin, new_ticket, dict_stars)

        # duplication was found, pick data from measures as evidence
        df_duplicates = df_sms.iloc[[y_idx_true]]
        df_duplicates = _restyle_dataframe(df_duplicates, metrics)
        
        logging.debug('--- df_duplicates [%s] ---'%new_ticket)
        logging.debug(df_duplicates.head())
        logging.debug('----------------------------------')
        
        return dict_stars, df_duplicates, y_true_pred_proba

def _create_star_origin(origin_ticket:str, dict_stars:Dict[str, str]) -> dict:
    '''
        Support function to create a new star since a new origin is detected.


        Parameters
        ----------
        origin_ticket: str
            Origin ticket ID

        dict_stars: dict
            Dictionary of origin and stars

        Returns
        -------
        dict_stars: dict
            Updated dictionary of origin and stars
    '''

    logging.debug('Creating new star with origin %s'%origin_ticket)
    G = nx.DiGraph()
    G.add_node(origin_ticket)
    dict_stars[origin_ticket] = G
    return dict_stars

def _get_origin(df_duplicates:pd.DataFrame, df_sms:pd.DataFrame, list_y_idx_true:List[int], metrics:List[str]) -> tuple:
    '''
        Support function to select the origin for a new duplicate.


        Parameters
        ----------
        df_duplicates: pd.DataFrame
            DataFrame of origins catalog and new ticket 

        df_sms: pd.DataFrame
            Dataframe with just similarities for each pair of tickets
        
        list_y_idx_true: list
            List of indexes of the possible origins

        metrics: list
            List of metrics name used to evaluate sentence similarity

        Returns
        -------
        origin: str
            Selected origin for the duplicate

        y_idx_true: str
            Index of the selected origin
    '''

    if len(list_y_idx_true) == 1:
        origin = df_duplicates.iat[0, 0]
        y_idx_true = list_y_idx_true[0]
    else:
        logging.warning('Multiple eligible origins are found: \n%s'%df_sms[metrics])
        # in this case multiple origins are eligible to be The origin
        # for the curernt ticket and this cannot be possible.
        # for each case (i.e. each eligible origins), compute the mean
        # of the metrics and keep the origin with the highest value as
        # origin for the current ticket

        logging.debug('Selecting the mean origin...')
        y_idx_true = df_sms[metrics].mean(axis=1).argmax()

        origin = df_sms.iat[y_idx_true, 1]
        logging.debug('Selected origin: %s'%origin)
    
    return origin, y_idx_true

def _store_in_star_origin(origin:str, new_ticket:str, dict_stars:Dict[str, str]) -> dict:
    '''
        Support function to store a new duplicate to an existing star.


        Parameters
        ----------
        origin: str
            Origin ticket ID

        new_ticket: str
            Duplicate ticket ID
        
        dict_stars: dict
            Dictionary of origin and stars

        Returns
        -------
        dict_stars: dict
            Updated dictionary of origin and stars
    '''

    # store current ticket in origin-star
    try:
        G = dict_stars[origin]
        G.add_edge(new_ticket, origin)
    except Exception as e:
        logging.info('ALERT: %s not in dict_stars'%origin)
        logging.info('new ticket: %s'%new_ticket)
        raise e # TODO: need to decide how to deal with this case
    return dict_stars

def _restyle_dataframe(df_duplicates:pd.DataFrame, metrics:List[str]) -> pd.DataFrame:
    '''
        Support function to restyle the duplicate dataframe.


        Parameters
        ----------
        df_duplicates: pd.DataFrame
            DataFrame of origins catalog and new ticket 

        metrics: list
            List of metrics name used to evaluate sentence similarity

        Returns
        -------
        df_duplicates: pd.DataFrame
            Re-styled dataframe of origins catalog and new ticket
    '''

    df_duplicates = df_duplicates.drop('duplication', axis=1). \
        rename(columns={df_duplicates.columns[0]: df_duplicates.columns[1], 
        df_duplicates.columns[1]: df_duplicates.columns[0]}). \
        sort_values(metrics, ascending = [True]*len(metrics))
    return df_duplicates

def store_recognize_duplicates(df:pd.DataFrame, dict_stars:Dict[str, str], dict_recognized_duplicates:Dict[str, pd.DataFrame]) -> None:
    '''
        This function is used to store duplicates and stars in excel files.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with just valid tickets

        dict_stars_filtered: dict
            Dictionary with duplicates and origins

        dict_recognized_duplicates: dict
            Dictionary of recognized duplicates with their dataframe

        Returns
        -------
        None
    '''

    from core.data_manipulation.data_handlers.SourceHandler import SourceHandler
    logging.info('\n##### STORE RECOGNIZED DUPLICATES #####\n')

    df_origins = df.incidentid.tolist()
    df_origins = [origin for origin in df_origins if origin in dict_stars.keys()]

    df_stars = {}
    for origin in df_origins:
        duplicates = list(dict_stars[origin].nodes)[1:]
        df_stars[origin] = duplicates
            
    df_stars = pd.DataFrame.from_dict(df_stars, orient='index')
    logging.debug('----- DF STARS ----\n%s\n-----------------'%df_stars)

    df_stars = df_stars.transpose()

    list_df_duplicates = list(dict_recognized_duplicates.values())
    if all(df is None for df in list_df_duplicates):
        log.warning('WARNING: No duplicates were found.')
        with pd.ExcelWriter(backup_location + destination, engine='openpyxl') as writer:
            df_stars.to_excel(writer, sheet_name='origin_stars', index=False)
        return None

    df_new_duplicates = pd.concat(list_df_duplicates)
    df_new_duplicates = df_new_duplicates.dropna()

    try:
        df_old_duplicates = pd.read_excel(backup_location + destination, sheet_name='duplicates')
        df_duplicates = pd.concat([df_new_duplicates, df_old_duplicates])
    except Exception as e:
        df_duplicates = df_new_duplicates
    
    df_duplicates = df_duplicates.drop('duplication', axis=1). \
        rename(columns={df_duplicates.columns[0]: df_duplicates.columns[1], 
        df_duplicates.columns[1]: df_duplicates.columns[0]})

    df_duplicates.insert(0,'date', len(df_duplicates)*[TODAY])
    
    df_duplicates.insert(len(df_duplicates.columns), 'duplication', 
                         len(df_duplicates)*['TO-BE-CONFIRMED'])

    with pd.ExcelWriter(backup_location + destination, engine='openpyxl') as writer:
        df_duplicates.to_excel(writer, sheet_name='duplicate_pairs', index=False)
        df_stars.to_excel(writer, sheet_name='origin_stars', index=False)

    source_handler = SourceHandler()
    source_handler.connect('DEVO', 'lab_dlb_eucso_uni_sla.itsm_duplicate_incidents')

    """
    CREATE TABLE lab_dlb_eucso_uni_sla.itsm_duplicate_incidents (
        computation_date TIMESTAMP,  
        duplicate_incidentid STRING,
        origin_incidentid STRING,
        JCD	FLOAT,
        DICE FLOAT,	
        LCS	FLOAT,
        SWO	FLOAT,
        TF_IDF FLOAT,
        GLOVE FLOAT,
        W2VEC FLOAT,
        WO FLOAT,
        JW FLOAT,
        BST FLOAT,
        duplication STRING,

        PRIMARY KEY (origin_incidentid, duplicate_incidentid)
    );
    """

    source_handler.store(df_duplicates)

def update_catalog(dict_stars:Dict[str, str], df_2_weeks_origins:pd.DataFrame) -> pd.DataFrame:
    '''
        This function is used to updated the catalog.

        Parameters
        ----------
        dict_stars_filtered: dict
            Dictionary with origins and stars

        df_2_weeks_origins: pd.DataFrame
            Catalog dataframe

        Returns
        -------
            pd.DataFrame
            Filter the catalog of last 2 weeks by taking just the origins
    '''

    return df_2_weeks_origins.loc[df_2_weeks_origins.incidentid. \
            isin(list(dict_stars.keys()))]

def post_production_analysis() -> None:
    from core.utils.plots.models_plots import compare_prediction_prob_distributions
    compare_prediction_prob_distributions()
