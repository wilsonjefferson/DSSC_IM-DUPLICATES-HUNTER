import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import joblib
import logging
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity


tqdm.pandas(desc='Eval sim range')
log = logging.getLogger('datasetBuilderLogger')

from ConfigNameSpace import MAIN_TRAIN
train_backup_location = MAIN_TRAIN.backup_location + 'tfidf/'
datasetbuilder_backup_location = MAIN_TRAIN.backup_location + '/step_construct_dataset_original/datasetbuilder_tmp_files/'

## PRE-TRAINED VECTORIZERS ##
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = joblib.load(train_backup_location + 'tfidf_pretrained_vectorizer.joblib')
#############################


class DatasetBuilder:
    '''
        This is a support class to build train/test datasets for the ML models.

        Attributes
        ----------
        df: pd.DataFrame
            Dataframe with all the row information

        true_duplicates_dict: dict
            Dictionary with the spotted pair (origin, duplicate) from the agents

        quasi_true_duplicates_dict: dict
            Artificial collection of duplicates created from a true duplicates
            but they are rephrased using the PharaGenerator model
        
        quasi_false_duplicates_dict: dict
            Artificial collection of duplicates created from a false duplicates
            but they are rephrased using the PharaGenerator model

        false_duplicates_dict: dict
            Dictionary with false pair of (origin, duplicate)
    '''

    def __init__(self, df:pd.DataFrame, duplicates_dict:Dict[str, str], location:str):
        self.df = df
        self.true_duplicates_dict = duplicates_dict
        self.quasi_true_duplicates_dict = None
        self.quasi_false_duplicates_dict = None
        self.false_duplicates_dict = None
        self.df_dataset = None
        self.df_artificial = None
        self.location = location

    def build_dataset(self, type:str) -> pd.DataFrame:
        '''
            The purpose of this method is to build the dataset of duplicate tickets,
            it will be composed by origin tickets and duplicate tickets; where the
            pairs (origin, duplicate) are classified as real duplicates (label 1), 
            or no duplicates (label 0).

            Parameters
            ----------
            destinations: list
                List of location(s), in case it is required to build multiple different
                dataset. At least one item in the list is required.

                Note:
                The benefit to define the list of destination inside the method
                is that each dataframes are sharing the duplication=1 portion, i.e.
                the part of the dataset referring to the recognized duplicates tickets
                by the agents.

            Returns
            -------
            list_df_datasets: list
                List of built dataframe datasets
        '''
        
        # define set of columns for the dataset (distinguish between origin and duplicate)
        log.debug('Defining set of columns for the dataset')
        dataset_columns = self.build_columns(True)
        
        # define real dataset (i.e. from duplicates recognized by agents) and 
        # false dataset (i.e. randomly created).
        # true dataset part has duplication field = 1 (yes duplication)
        # false dataset part has duplication field = 0 (no duplication)
        log.debug('Building data for the dataset')
        
        if type == 'original':
            self._construct_duplicates_dict(False)
            true_data = self.build_data(self.true_duplicates_dict, 'true', 1)
            false_data = self.build_data(self.false_duplicates_dict, 'false', 0)
            log.debug('true data array shape: (%s, %s)'%true_data.shape)
            log.debug('false data array shape: (%s, %s)'%false_data.shape)

            # construct dataset dataframe
            self.df_dataset = np.concatenate((true_data, false_data), axis=0)
            self.df_dataset = pd.DataFrame(self.df_dataset, columns = dataset_columns)
            dataset = self.df_dataset
        elif type == 'artificial':
            self._construct_duplicates_dict(True)
            quasi_true_data = self.build_data(self.quasi_true_duplicates_dict, 'quasi_true', 1)
            quasi_false_data = self.build_data(self.quasi_false_duplicates_dict, 'quasi_false', 1)
            log.debug('quasi-true data array shape: (%s, %s)'%quasi_true_data.shape)
            log.debug('quasi-false data array shape: (%s, %s)'%quasi_false_data.shape)

            # construct dataset dataframe
            self.df_artificial = np.concatenate((quasi_true_data, quasi_false_data), axis=0)
            self.df_artificial = pd.DataFrame(self.df_artificial, columns = dataset_columns)
            dataset = self.df_artificial
        else:
            raise Exception('dataset type %s is not acceptable, please choose between original and artificial'%type)
        
        return dataset

    def _construct_duplicates_dict(self, build_artificial:bool) -> None:
        '''
            This is a support method, it is used internally the class to construct
            the duplicates dictionary. In particular, it can be divided in 3 parts:
            - construct false duplicates dictionary
            - construct quasi false duplicates dictionary
            - construct quasi true duplicates dictionary

            Those dictionaries are composed as (duplicate, origin) and have the same
            number of items (in orther to construct a balanced dataset), the number of 
            true duplicates in the true_duplicates_dict.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''

        if len(self.true_duplicates_dict) == 0: # TODO: Input Validation
            # TODO: custom exception required
            raise log.critical('No pair of true duplicates were found')

        true_duplicates = list(set(self.true_duplicates_dict.keys())) + list(set(self.true_duplicates_dict.values()))
        log.info('List of true duplicates (origin+duplicates): %s'%len(true_duplicates))

        other_tickets = list(set(self.df['incidentid'].tolist()) - set(true_duplicates))
        log.info('List of others: %s'%len(other_tickets))

        if len(other_tickets) == 0: # TODO: Input Validation
            # TODO: custom exception required
            raise log.critical('No duplicates tickets to be used for negative category was provided')       

        # list of all the tickets recognized by the agents (origins + duplicates)
        # and list of all the other tickets available
        lists = [true_duplicates, other_tickets]
        
        # recover intermidiate back-up or run again the class method
        if build_artificial:
            log.info('Constructing duplicates artificially')
            location = self.location + 'step_construct_dataset_artificial/'

            try:
                with open(location + 'DatasetBuilder_quasi_false.pickle', 'rb') as handle:
                    self.quasi_false_duplicates_dict = pickle.load(handle)
                log.info('Valid quasi false duplicates backup found')
            except Exception as e:
                self.quasi_false_duplicates_dict = self._construct(lists, .55, .7)
                with open(location + 'DatasetBuilder_quasi_false.pickle', 'wb') as handle:
                    pickle.dump(self.quasi_false_duplicates_dict, handle, 
                                protocol=pickle.HIGHEST_PROTOCOL)

            try:
                with open(location + 'DatasetBuilder_quasi_true.pickle', 'rb') as handle:
                    self.quasi_true_duplicates_dict = pickle.load(handle)

                df_quasi_true_tmp = pd.read_excel(location + 'DatasetBuilder_df_quasi_true_tmp.xlsx')
                
                log.info('Valid quasi true duplicates backup found')
            except Exception as e:
                self.quasi_true_duplicates_dict = self._construct_quasi_true(other_tickets, .65, .8)
                with open(location + 'DatasetBuilder_quasi_true.pickle', 'wb') as handle:
                    pickle.dump(self.quasi_true_duplicates_dict, handle, 
                                protocol=pickle.HIGHEST_PROTOCOL)
                    self.df[self.df.incidentid.str.contains('QT')].to_excel(
                        location + 'DatasetBuilder_df_quasi_true_tmp.xlsx', index=False)
            else:
                # join along rows the df and the back of quasi-true ticket pairs
                self.df = pd.concat([self.df, df_quasi_true_tmp], ignore_index=True, sort=False)
                self.df.to_excel(location + 'DatasetBuilder_df_tmp.xlsx', index=False)
        else:
            location = self.location + 'step_construct_dataset_original/'
            try:
                with open(location + 'DatasetBuilder_false.pickle', 'rb') as handle:
                    self.false_duplicates_dict = pickle.load(handle)
                log.info('Valid false duplicates backup found')
            except Exception as e:
                self.false_duplicates_dict = self._construct(lists, .0, .5)
                with open(location + 'DatasetBuilder_false.pickle', 'wb') as handle:
                    pickle.dump(self.false_duplicates_dict, handle, 
                                protocol=pickle.HIGHEST_PROTOCOL)
    
    def _construct_quasi_true(self, tickets:list, min:float, max:float) -> dict:
        # NOTE: Function to be tested and corrected in case of bugs
        
        from core.utils.PharaGenerator import PharaGenerator
        paraphraser = PharaGenerator()

        n_repetition = len(tickets)

        id_candidates = np.random.choice(tickets, size=n_repetition, replace=False)
        texts = self.df[self.df.incidentid.isin(id_candidates)].text_cleaned
        texts_phara, _ = texts.apply(paraphraser.get_phrase, (min, max))

        ids_data = np.empty(shape=(n_repetition, 2), dtype=np.dtype('<U10'))
        np.copyto(ids_data[:,0], id_candidates)
        np.copyto(ids_data[:,1], id_candidates)
        ids_data[:,1] = np.char.replace(ids_data[:,1], 'IM', 'QT')
        
        df_ids_data = pd.DataFrame(ids_data, columns=['duplicate', 'origin'])
        data = pd.concat([df_ids_data, texts_phara], axis = 1)
        data.apply(self._insert_fake_row, axis = 1)

        return dict(ids_data)
    
    def _construct(self, tickets:List[List[str]], minm:float, maxm:float) -> dict:
        '''
            The purpose of this method is to create a dictionary of false pairs of
            duplicate tickets.
            In this method the following subsets of tickets are used:
            - true_duplicates = list of the tickets recognized as duplicates (or origin)
            - other_tickets = list of all the other tickets not in true_duplicates
            Those are used as sources of tickets to define the fake duplicates dictionary.

            Parameters
            ----------
            tickets: List[List[str]]
                List of tickets, this list is composed of two sub-lists

            minm: float
                Minimum value of similarity

            maxm: float
                Maximum value of similarity

            Returns
            -------
            fake_duplicates_dict: dict
                Dictionary of fake duplicates (duplicate -> origin)
        '''

        nrows = len(self.true_duplicates_dict)
        log.debug(f'nrows: {nrows}')
        log.debug(f'len tickets[0]: {len(tickets[0])}')
        log.debug(f'len tickets[1]: {len(tickets[1])}')

        from math import factorial as fact

        _count_permutation = lambda l: int(fact(len(l)) / fact(len(l)-2))
        
        perm = {_count_permutation(tickets[0]):'DD', 
                _count_permutation(tickets[1]):'OO', 
                _count_permutation(tickets[0]+tickets[1]):'DO'}
        tickets_space = {'DD':tickets[0], 'OO':tickets[1], 'DO':tickets[0]+tickets[1]}

        n_repetation = nrows + int(nrows/2)
        log.debug(f'number of repetition: {n_repetation}')

        first_row, last_row = (0, 0)
        
        # NOTE: !! WARNING !!
        # np.dtype('<U10') means ticket is at most composed by 10 characters
        # as example:
        # len('IM123456') = 8 -> accepted
        # len('IM12345678') = 10 -> accepted
        # len('IM1234567890') = 12 -> TRUNCATED --> IM12345678
        # if you want to increase the number of characters allowed change from
        # <U10 to <U12 or <U14 as your preference
        ticket_n_chars = len(tickets[0][0])
        if ticket_n_chars == 0:
            raise log.exception('sample ticket is not valid, check df dataframe')
        dtype = '<U'+str(ticket_n_chars) # automatically adjusted
        ids_data = np.empty(shape=(nrows, 2), dtype=np.dtype(dtype))

        seen_indexes = {'DD':[], 'OO':[], 'DO':[]}

        while (ids_data == ['','']).any():

            tmp_data, seen_indexes = self._random_choice(tickets_space, perm, seen_indexes, n_repetation)

            mask_not_unique_keys = self._not_unique_keys(tmp_data[:,0], ids_data[:,0])
            mask_duplicate_rows = self._duplicate_rows(tmp_data)
            mask_same_id = self._same_ids(tmp_data)
            mask_out_sim_range = self._out_sim_range(tmp_data, minm, maxm)
            mask_already_seen = self._seen_rows(tmp_data, ids_data)

            log.info('mask_not_unique_keys shape: %s'%mask_not_unique_keys.shape)
            log.info('mask_duplicate_rows shape: %s'%mask_duplicate_rows.shape)
            log.info('mask_same_id shape: %s'%mask_same_id.shape)
            log.info('mask_out_sim_range shape: %s'%mask_out_sim_range.shape)
            log.info('mask_already_seen shape: %s'%mask_already_seen.shape)

            mask = mask_not_unique_keys | mask_duplicate_rows | mask_same_id | mask_out_sim_range | mask_already_seen
            log.info('mask shape: %s'%mask.shape)

            ### store mask matrix ###
            t_masks = (mask_duplicate_rows, mask_same_id, mask_out_sim_range, \
                                  mask_already_seen, mask, ~mask)
            tmp = np.concatenate(t_masks).reshape((-1, len(t_masks)), order='F')
            pd.DataFrame(tmp, columns=['duplicate_rows', 'same_id', 'out_sim_range', \
                                       'already_seen', 'mask', 'not_mask']).to_excel(
                datasetbuilder_backup_location + 'mask.xlsx')
            
            pd.DataFrame(tmp_data, columns=['t1', 't2']).to_excel(
                datasetbuilder_backup_location + 'tmp_data.xlsx')
            #########################
            
            data = tmp_data[~mask]
            log.info('tmp_data shape: (%s, %s)'%data.shape)
            last_row += data.shape[0]
            log.info(f'first row: {first_row} last row: {last_row} ' +
                      f'nrows: {nrows} n_repetition: {n_repetation}')

            if last_row <= nrows: # store outcome in no-temporal array
                np.copyto(ids_data[first_row:last_row,:], data[:,:])
            else:
                np.copyto(ids_data[first_row:,:], data[:nrows-first_row,:])
                last_row = nrows

            ### store ids_data matrix ###
            pd.DataFrame(ids_data, columns=['t1', 't2']).to_excel(
                datasetbuilder_backup_location + 'ids_data.xlsx')
            #############################

            first_row = last_row
            n_repetation = nrows - last_row
            n_repetation += int(n_repetation/2)
            log.debug(f'number of repetition: {n_repetation}')

        log.info(f'ids_data fulfilled with {len(ids_data)} pairs of duplicates and origins')
        return dict(ids_data)
    
    def _random_choice(self, tickets:Dict[str, list], perm:Dict[str,list], seen_indexes:Dict[str,list], n_repetition:int):
        '''
        Consider the following sets:
        - D = {list of duplicates and origins}
        - O = {list of all other incidents id}
        - DO = OD = {union of D and O lists}

        - DD_2 = {list of 2 elements as permutation of D}
        - OO_2 = {list of 2 elements as permutation of O}
        - DO_2 = OD_2 = {list of 2 elements as permutation of DO}

        #D < #O < #DO and #DD_2 << #OO_2 < #DO_2

        The objective is to create a dataset where we have candidates equaly
        selected from DD_2, OO_2 and DO_2 so each set has 1/3 of probability
        to be choose; then a single element of the choosen set is randomly pick.

        Howerver, we cannot instantiate the 3 sets in the RAM otherwise we would'
        need more than 1 tera of memory. So, no one list is instantiated but we
        know that a permutation is an ordered set, i.e.

        set_0 = {A, B, C, D, E, F, G}

        permutation = {
            [AB, AC, AD, AE, AF, AG] = subset S_A where first element is A
            [BA, BC, BD, BE, BF, BG] = subset S_B where first element is B
            [CA, CB, CD, CE, CF, CG] = subset S_C where first element is C
            [DA, DB, DC, DE, DF, DG] = subset S_D where first element is D
            [EA, EB, EC, ED, EF, EG] = subset S_E where first element is E
            [FA, FB, FC, FD, FE, FG] = subset S_F where first element is F
            [GA, GB, GC, GD, GE, GF] = subset S_G where first element is G
        }

        #permutation = #set_0 * #S_tot = #set_0 * (#set_0 -1) = 42

        Each subset S_i contains a certain number of INDEXED elements of the 
        permutation set, i.e.

        S_A(idx) = index{0, 1, 2, 3, 4, 5}

        So, what is randomly choosen is not the pair of elements but the position
        i.e. the index; having the index we can assess on which subset S_i it is
        contained (and what is the first element of the pair since it is fixed).

        Let's say we choose idx = 17 (that is between 0 and 41).

        17/(#S_tot) = 17/6 = 2.833333... -> 2 integer and 0.8333 mantissa

        idx 2 into set_0 is C so, the first element is C and we have to consider
        the subset S_C = [CA, CB, CD, CE, CF, CG], one of these elements is in
        position 17 in permutation set. We need to assess the second element:

        0.8333/(1/(#set_0 -1)) = 0.8333/(1/6) = 0.8333/0.1666 = 5 more or less

        idx 5 into S_C is CG.

        So, given a random index 17 we can assess on what is the corrisponding
        element in permutation set, i.e. CG.

        This logic is here developed. and repeated n_repetition


        Parameters
        ----------
        tickets: Dict[str, list]
            Dictionary of {'DD':tickets[0], 'OO':tickets[1], 'DO':tickets[0]+tickets[1]}

        perm: Dict[str,list]
        Dictionary of {_count_permutation(tickets[0]):'DD', 
                _count_permutation(tickets[1]):'OO', 
                _count_permutation(tickets[0]+tickets[1]):'DO'}
            

        seen_indexes: Dict[str,list]
            Dictionary of already seen indexes for each set
        
        n_repetition:int
            Number of random indexes to be picked
        
        Returns
        -------
        selected_elements: np.array
            2D array of the elements

        seen_indexes: Dict[str,list]
            Dictionary of already seen indexes for each set
        '''
        
        from random import randint, choice
        from math import modf

        i = 0
        selected_elements = []
        while i < n_repetition:

            n_perm = choice(list(perm.keys()))
            log.debug(f'n_perm: {n_perm}')

            set_perm = perm[n_perm]
            log.debug(f'n_perm: {set_perm}')

            seen = seen_indexes[set_perm]
            all_tickets = tickets[set_perm]

            r = randint(0, n_perm)
            while r in seen:
                r = randint(0, n_perm)
            log.debug(f'index: {r}')
            seen.append(r)

            frac = r/(len(tickets[set_perm])-1)
            log.debug(f'frac: {frac}')

            scd_id, frt_id = modf(frac)
            log.debug(f'frt_id: {frt_id} scd_id: {scd_id}')

            frt_id = int(frt_id)
            scd_id = int(scd_id/(1/(len(tickets[set_perm])-1)))
            log.debug(f'Adjustment float to int- frt_id: {frt_id} scd_id: {scd_id}')

            if frt_id >= len(all_tickets) or scd_id >= len(all_tickets):
                log.warning(f'Exceeded range - frt_id: {frt_id} scd_id: {scd_id}')
                continue
            elif frt_id == scd_id:
                log.warning(f'Same idexes - frt_id: {frt_id} scd_id: {scd_id}')
                continue

            frt_id, scd_id = all_tickets[frt_id], all_tickets[scd_id]
            log.debug(f'frt_id: {frt_id} scd_id: {scd_id}')

            selected_elements.append((frt_id,scd_id))
            i += 1

        log.debug('length selected_elements: %s'%len(selected_elements))

        selected_elements = np.array(selected_elements).reshape((-1, 2))
        return selected_elements, seen_indexes

    def _duplicate_rows(self, array:np.array) -> np.array:
        '''
            Support function to find duplicates 
            in array and keep just the first occurance.

            Parameters
            ----------
            array: np.array
                array of tickets

            Returns
            -------
                np.array
                Array of unique tickets
        '''

        return pd.DataFrame(array).duplicated(keep='first').to_numpy()

    def _not_unique_keys(self, array:np.array, ids_key_data:np.array) -> np.array:
        '''
            Support function to find duplicates 
            in ids_key_data and array and keep just the first occurance.

            Parameters
            ----------
            array: np.array
                array of tickets

            ids_key_data:np.array
                array of tickets

            Returns
            -------
                np.array
                Array of unique tickets
        '''

        tmp =  np.concatenate((ids_key_data, array), axis=0)
        return pd.DataFrame(tmp).duplicated(keep='first').to_numpy()[len(ids_key_data):]
    
    def _same_ids(self, array:np.array) -> np.array:
        '''
            Support function to find pairs of tickets with same id.

            Parameters
            ----------
            array: np.array
                Array of tickets

            Returns
            -------
                np.array
                Array of bool
        '''
        
        return (array[:,0] == array[:,1])
    
    def _out_sim_range(self, array:np.array, minm:float, maxm:float) -> np.array:
        '''
            Support function to compute verifyif pairs of tickets have 
            similarity value outside a given similarity range.

            Parameters
            ----------
            array: np.array
                Array of tickets

            minm: float
                Minimum similarity value

            maxm: float
                Maximum similarity value

            Returns
            -------
                np.array
                Array of outside similarity range
        '''

        t1 = pd.DataFrame(array[:,0], columns=['incidentid'])
        t2 = pd.DataFrame(array[:,1], columns=['incidentid'])
        log.debug('t1 rows: %s t2 rows: %s'%(t1.shape[0], t2.shape[0]))

        tmp = pd.DataFrame(array, columns=['t1_incidentid','t2_incidentid'])
        tmp.to_excel(datasetbuilder_backup_location + 'array_in_out_sim_range.xlsx')

        df_tmp = self.df[['incidentid', 'text_cleaned']]
        t1 = t1.merge(df_tmp, on='incidentid', how='left', sort=False).add_prefix('t1_')
        t2 = t2.merge(df_tmp, on='incidentid', how='left', sort=False).add_prefix('t2_')
        log.debug('t1 rows: %s t2 rows: %s'%(t1.shape[0], t2.shape[0]))
        
        texts = pd.concat([t1, t2], axis=1)
        log.debug('texts - dataframe shape: (%s, %s)'%texts.shape)
        
        texts = texts[:array.shape[0]] # NOTE: adjustment required

        texts[['sim_range', 'out_sim_range']] = texts[['t1_text_cleaned', 't2_text_cleaned']].progress_apply( \
                                    self._out_range, args=(minm, maxm), axis=1, result_type='expand')
        
        texts.to_excel(datasetbuilder_backup_location + 'texts_in_out_sim_range.xlsx')

        if (texts['out_sim_range'] == True).all():
            log.warning('WARNING: No a single pair is out of similarity range!')
        
        return texts.out_sim_range.to_numpy()

    def _out_range(self, tickets_pair:pd.Series, minm:float, maxm:float) -> bool:
        '''
            This is a support method, it is used internally the class to evaluate 
            if the similarity score of a pair of tickets is in a valid range of values.

            Parameters
            ----------
            ticket1: str
                ticket text

            ticket2: str
                ticket text

            min: float
                minimum value

            max: float
                maximum value

            Returns
            -------
                bool
                True if score is in the range, False otherwise
        '''
        
        sim = self._similiarity_score(tickets_pair)
        check_condition = sim < minm or sim > maxm
        return (sim, check_condition)

    def _similiarity_score(self, tickets_pair:pd.Series) -> float:
        '''
            This is a support method, it is used internally the class to evaluate 
            the similarity score of a pair of text using tf_idf_similarity.

            Parameters
            ----------
            ticket1: str
                ticket text

            ticket2: str
                ticket text

            Returns
            -------
            sim: float
                Similarity score
        '''
        from core.metrics_and_models.benchmarks import tf_idf_similarity
        
        try:
            tfidf_matrix = tfidf.transform(tickets_pair)
            embedding1, embedding2 = tfidf_matrix[0], tfidf_matrix[1]
            sim = cosine_similarity(embedding1, embedding2)[0][0]
        except ValueError as e:
            sim = 0
            log.debug(f'{tickets_pair}: {e}')
        
        return sim

    def _seen_rows(self, array:np.array, ids_data:np.array) -> np.array:
        '''
            Support function to check if a row was already seen

            Parameters
            ----------
            array: np.array
                Array of tickets

            ids_data: np.array
                Array of tickets

            Returns
            -------
                np.array
                    Array of bool
        '''

        return (array==ids_data[:,None]).all(-1)[0]

    def _insert_fake_row(self, ticket:str, new_ticket:str, sentence:str) -> None:
        '''
            This is a support method, it is used internally the class to insert artificial
            data in the incident Dataframe. This method is used just to construct the quasi
            true duplicates dictionary, since those duplicates are actually paraphrases of the
            true duplicates.

            Parameters
            ----------
            ticket: str
                ticket id

            new_ticket: str
                artificial ticket id

            sentence: str
                ticket text

            Returns
            -------
            None
        '''
        df_tmp = self.df.loc[self.df['incidentid'] == ticket]
        df_tmp = df_tmp.reset_index(drop=True)
        df_tmp.loc[0, 'incidentid'] = new_ticket
        df_tmp.loc[0, 'text_cleaned'] = sentence
        df_tmp = pd.Series(df_tmp.values[0], df_tmp.columns)
        self.df.loc[len(self.df.index)] = df_tmp
    
    def build_data(self, duplicates_dict:Dict[str, str], duplication_category:str = None, duplication:bool = None) -> list:
        '''
            This method gather the raw data that will form the dataset.

            Parameters
            ----------
            df: pd.DataFrame
                DataFrame of the tickets

            duplicates_dict: dict
                Dictionary of duplicates (duplicate -> origin)

            duplication_category: str
                Category for the row data, in particular we can say:
                - true, for the true duplicates
                - quasi-true, for the almost true duplicates
                - quasi-false, for the almost false duplicates
                - false, for the false duplicates

            duplication: bool
                Decleare if the pair(s) of ticket(s) is/are duplicates or not

            Returns
            -------
            row_tickets: list
                List of valid pairs of tickets classified
        '''

        log.debug(f'duplication category: {duplication_category}')

        duplicates = pd.DataFrame(duplicates_dict.keys(), columns=['incidentid'])
        origins = pd.DataFrame(duplicates_dict.values(), columns=['incidentid'])
        log.debug('duplicates: %s origins: %s'%(duplicates.shape[0], origins.shape[0]))
        
        df_duplicates = self.df.merge(duplicates, on='incidentid', sort=False, how='right')
        df_origins = self.df.merge(origins, on='incidentid', sort=False, how='right')

        log.debug('df_duplicates: %s df_origins: %s'%(df_duplicates.shape[0], df_origins.shape[0]))

        categories = np.full((df_duplicates.shape[0], 1), duplication_category)
        duplications = np.full((df_duplicates.shape[0], 1), duplication)

        try:
            rows = np.concatenate((df_origins, df_duplicates, categories, duplications), axis=1)
        except ValueError as e:
            from ConfigNameSpace import MAIN_STAGE
            stage_backup_location = MAIN_STAGE.backup_location

            log.exception(f'df_origins len: {df_origins.shape}')
            log.exception(f'df_duplicates len: {df_duplicates.shape}')

            log.exception(f'categories len: {categories.shape}')
            log.exception(f'duplications len: {duplications.shape}')

            df_origins.to_excel(stage_backup_location+'df_origins.xlsx')
            df_duplicates.to_excel(stage_backup_location+'df_duplicates.xlsx')

            raise e
        else:
            return rows

    def build_columns(self, duplication:bool) -> list:
        '''
            Ths method set the headers for the dataset.

            Parameters
            ----------
            duplication: bool
                Decleare if the pair(s) of ticket(s) is/are duplicates or not

            Returns
            -------
            dataset_columns: list
                List of columns name for dataset
        '''

        origins_columns = ['origin_' + column for column in self.df.columns]
        duplicates_columns = ['duplicate_' + column for column in self.df.columns]
        dataset_columns = origins_columns + duplicates_columns
        if duplication:
            dataset_columns += ['duplication_category', 'duplication']
        return dataset_columns
    