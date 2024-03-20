import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict

import logging
log = logging.getLogger('modelTrainComponentsLogger')

from ConfigNameSpace import TODAY, MAIN_TRAIN, project_root
backup_location = MAIN_TRAIN.backup_location
make_plots = MAIN_TRAIN.make_plots

from core.data_manipulation.data_filtering import remove_no_valid_tickets, define_sentences
from core.utils.recovery_system import recovery_decorator


def import_data(source:str = 'lab_dlb_eucso_uni_sla.itsm_incidents', query:str = None) -> pd.DataFrame:
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
    log.info('\n\n##### IMPORT DATA #####\n')
    
    source_handler = SourceHandler()
    source_handler.connect('DISC', source)
    df_original = source_handler.get_data(query)
      
    return df_original

@recovery_decorator(['step_filter_invalid_tickets/filter_no_valid_tickets.xlsx'])
def filter_invalid_tickets(df_original:pd.DataFrame) -> pd.DataFrame:
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

    df, _ = remove_no_valid_tickets(df_original, ['description', 'title'])
    df['text'] = df.title + pd.Series(len(df)*['. ']) + df.description
    
    log.debug(df)
    return df

@recovery_decorator(['step_construct_duplicates_dict/stars_filtered.xlsx', 
                'step_construct_duplicates_dict/dict_stars_filtered.pickle'])
def construct_duplicates_dict(df:pd.DataFrame) -> tuple:
    '''
        Construct dictionary with duplicate tickets (key) and
        origin tickets (value).

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with just valid tickets

        Returns
        -------
        df: pd.DataFrame
            Dataframe with just valid tickets
        
        dict_stars_filtered: dict
            dictionary with duplicates and origins
    '''

    from core.data_manipulation.recognized_duplicates import search_recognized_duplicates
    log.info('\n\n##### CONSTRUCT DUPLICATES DICTIONARY #####\n')

    dict_stars_filtered, dict_tickets_filtered = search_recognized_duplicates(df)
    log.debug('dict_stars_filtered: %s\ndict_tickets_filtered: %s'%
                    (len(dict_stars_filtered), len(dict_tickets_filtered)))

    if len(dict_stars_filtered) == 0:
        raise Exception('dict_stars_filtered is empty, no stars found')

    return df, dict_stars_filtered

@recovery_decorator(['step_text_preprocessing/df_cleaned.xlsx', 
                    'step_text_preprocessing/dict_stars_filtered.pickle',
                    'step_text_preprocessing/dict_origin_duplicates.pickle'])
def text_preprocessing(df:pd.DataFrame, dict_stars_filtered:Dict[str, str]) -> tuple:
    '''
        Compute text preprocessing and filter dict_stars_filtered from
        no valid pairs.

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

        dict_origin_duplicates: dict
            Dictionary with {origin: list of duplicates}
    '''

    from core.data_manipulation.text_pre_processing import text_processing, soft_preprocess
    log.info('\n\n##### COMPUTE TEXT PREPROCESSING #####\n')

    df, _ = text_processing(df, soft_preprocess, ['text'])
    df = df.astype("string")
    df, _ = define_sentences(df, ['text_cleaned'])
    log.debug(df.head())

    # remove incidents from dict_stars_filtered no longer valid
    # after text-preprocessing routine
    list_no_valid_recognized_tickets = []
    ids = df.incidentid.tolist()
    for duplicate, origin in dict_stars_filtered.items():
        if duplicate not in ids or origin not in ids:
            list_no_valid_recognized_tickets.append(duplicate)

    for k in list_no_valid_recognized_tickets:
        dict_stars_filtered.pop(k, None)

    dict_origin_duplicates = {}
    for key, value in dict_stars_filtered.items():
        if value not in dict_origin_duplicates:
            dict_origin_duplicates[value] = [key]
        else:
            dict_origin_duplicates[value].append(key)   
    
    return df, dict_stars_filtered, dict_origin_duplicates

def text_descriptive_analysis(df:pd.DataFrame) -> None:
    from core.data_manipulation.text_analysis import text_analysis

    if make_plots:
        log.info('\n\n##### PERFORM TEXT ANALYSIS #####\n')
        text_analysis(df)

@recovery_decorator(['tfidf/tfidf_pretrained_vectorizer.joblib'])
def pretrain_tfidf(df:pd.DataFrame) -> None:
    """
    Pretrain a TF-IDF Vectorizer using the provided DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing text data to train the TF-IDF Vectorizer.

    Returns:
    None: The function saves the TF-IDF Vectorizer vocabulary to a JSON file.

    Note:
    This function fits a TF-IDF Vectorizer on the cleaned text data in the DataFrame
    and saves the resulting vocabulary to a JSON file. The TF-IDF Vectorizer is useful
    for transforming text data into numerical features based on term frequency-inverse
    document frequency (TF-IDF) weighting.

    Parameters
    ----------
        df: DataFrame 
            The DataFrame containing a column 'text_cleaned' with preprocessed text data.

    """

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()
    training_data = df.text_cleaned.tolist()

    tfidf.fit(training_data)

    # Get the feature names (vocabulary) from the vectorizer
    feature_names = tfidf.get_feature_names_out().tolist()

    # Save the vocabulary to a JSON file
    import json
    tfidf_vocabulary = backup_location + 'tfidf/tfidf_vocabulary.json'
    with open(tfidf_vocabulary, 'w') as f:
        json.dump(feature_names, f)
    
    return tfidf

def pretrain_fasttext(df:pd.DataFrame, dict_origin_duplicates:dict) -> None:
    """
    Pretrain a FastText model on a DataFrame with text data.

    Note:
    This function performs unsupervised training for FastText on incident text data. It prepares
    training data in the FastText format with custom labels for each origin and duplicates. The
    resulting model is saved to a binary file for later use.

    Parameters
    ----------
        df: pd.DataFrame
            DataFrame containing a column 'text_cleaned' with preprocessed text data.

        dict_origin_duplicates: dict
            Dictionary where keys are origin IDs, and values are lists of duplicate incident 
            IDs associated with each origin.
    """

    import os
    import fasttext

    fasttext_pretrained_model = backup_location + 'fasttext/fasttext_pretrained_model.bin'
    if os.path.exists(fasttext_pretrained_model):
        return None

    fasttext_training_data = backup_location + 'fasttext/fasttext_train_custom_labels.txt'
    if not os.path.exists(fasttext_training_data):
        log.info('fasttext training data does not exist and need to be prepared.')
        training_data = []
        for origin, duplicates in dict_origin_duplicates.items():
            label = '__label__'+str(origin)+' '
            docs = df.loc[df['incidentid'].isin(duplicates), 'text_cleaned'].tolist()
            docs = [label+str(text) for text in docs]
            training_data += docs

        with open(fasttext_training_data, 'w', encoding='utf-8') as f:
            for line in training_data:
                f.write(line + '\n')
        log.info(f'fasttext training data prepared and stored in {fasttext_training_data}')
        
    log.info('unsupervised training for fasttext model on IM data in-progress...')
    model = fasttext.train_unsupervised(fasttext_training_data, model='cbow')
    model.save_model(fasttext_pretrained_model)
    log.info('fasttext pretrained model saved.')
    
    return model

@recovery_decorator(['doc2vec/doc2vec_pretrained_model.model'])
def pretrain_doc2vec(df:pd.DataFrame) -> None:
    """
    Pretrain a Doc2Vec model on a DataFrame with text data.

    Note:
    This function tags and preprocesses the text data for Doc2Vec training and trains
    a Doc2Vec model. The resulting model and vocabulary are saved to files for later use.

    Parameters
    ----------
        df: pd.DataFrame
            DataFrame containing a column 'text_cleaned' with preprocessed text data.
    """
    
    import gensim
    from gensim.models.doc2vec import TaggedDocument
    from core.utils.Ticket import Ticket

    corpus = df.text_cleaned.tolist()
    log.debug('Tagging Doc2Vec corpus in-progress...')
    corpus_tagged = [TaggedDocument(Ticket(doc).tokens_no_stop, [i]) for i, doc in enumerate(corpus)]

    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

    log.info('Training of Doc2Vec in-progress...')
    with tqdm(total=model.epochs, desc="Training Doc2Vec Model") as pbar:
        for epoch in range(model.epochs):
            model.build_vocab(corpus_tagged, progress_per=1)
            model.train(corpus_tagged, total_examples=model.corpus_count, epochs=1)
            pbar.update(1)  # Update progress bar manually since gensim doesn't provide progress
    
    with open(backup_location + "doc2vec/doc2vec_vocab.txt", "w", encoding="utf-8") as f:
        for word in model.wv.vocab:
            f.write(word + "\n")
    log.info(f'Doc2Vec vocabulary stored in {backup_location + "doc2vec/doc2vec_vocab.txt"}')

    model.save(backup_location + "doc2vec/doc2vec_pretrained_model.model")
    log.info('Doc2Vec pretrained model saved.')

    return None

@recovery_decorator(['step_construct_dataset_original/dataset.xlsx'])
def construct_dataset_original(df:pd.DataFrame, dict_stars_filtered:Dict[str, str]) -> pd.DataFrame:
    '''
        Construct the dataset.

        Parameters
        ----------
        dataset_builder: DatasetBuilder
            Object to build the dataset for the model(s)

        Returns
        -------
        df_dataset: pd.DataFrame
            Dataframe to train/test (and stress test) the model
    '''

    from core.data_manipulation.DatasetBuilder import DatasetBuilder

    log.info('\n\n##### CONSTRUCT ORIGINAL DATASET #####\n')
    dataset_builder = DatasetBuilder(df, dict_stars_filtered, backup_location)
    df_dataset = dataset_builder.build_dataset('original')
    return df_dataset

@recovery_decorator(['step_construct_dataset_artificial/stress_dataset.xlsx'])
def construct_dataset_artificial(df:pd.DataFrame, dict_stars_filtered:Dict[str, str]) -> pd.DataFrame:
    '''
        Construct the dataset.

        Parameters
        ----------
        dataset_builder: DatasetBuilder
            Object to build the dataset for the model(s)

        Returns
        -------
        df_dataset: pd.DataFrame
            Dataframe to train/test (and stress test) the model
    '''
    
    from core.data_manipulation.DatasetBuilder import DatasetBuilder

    log.info('\n\n##### CONSTRUCT ARTIFICIAL DATASET #####\n')
    log.info(f'dict_stars_filtered len: {len(dict_stars_filtered)}')
    dataset_builder = DatasetBuilder(df, dict_stars_filtered, backup_location)
    df_dataset = dataset_builder.build_dataset('artificial')
    return df_dataset

@recovery_decorator(['step_compute_similarities/data_measure.xlsx', 
                    'step_compute_similarities/data_measure_metrics.pickle'])
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
    log.info('\n\n##### COMPUTE SENTENCES METRICS #####\n')

    metrics, df_sms = compute_metrics(df_dataset,
        destination=backup_location+'step_compute_similarities/data_measure_complete.xlsx')
    
    log.debug(df_sms)
    return df_sms, metrics

@recovery_decorator(['dict_best_model.pickle', 'pred_probs/true_train_probs.joblib', 'pred_probs/true_test_probs.joblib'])
def explore_models(df_sms:pd.DataFrame, metrics:List[str]) -> object:
    '''
        Train and test all the available models to determine what
        is the best one.

        Parameters
        ----------
        df_sms: pd.DataFrame
            Dataframe with just similarities for each pair of tickets

        metrics: list
            List of metrics name used to evaluate sentence similarity

        Returns
        -------
        best_model: object
            Best model retrieved after the model exploration
    '''
    
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS

    from skopt import Space
    from skopt.space import Real, Integer, Categorical

    from core.metrics_and_models.TrainModelsRunner import TrainModelsRunner
    log.info('\n\n##### EXPLORE MODELS #####\n')

    X = np.array(df_sms.loc[:, metrics])
    y = np.array(df_sms.loc[:, 'duplication'])
    y = y.astype('int')

    sfs2 = SFS(RandomForestClassifier(random_state=0, n_estimators = 100), 
            k_features=2,
            forward=True,
            floating=False,
            scoring='accuracy',
            cv=10).fit(X, y)

    sfs4 = SFS(RandomForestClassifier(random_state=0, n_estimators = 100), 
            k_features=4, 
            forward=True,
            floating=False,
            scoring='accuracy',
            cv=10).fit(X, y)

    
    class_weights = {0: 1, 1: 10}
    hyperspaces_path = project_root + '/src/core/metrics_and_models/hyperspaces.yaml'

    ### SVM hyperparameters ###
    svm_param = {'kernel':'rbf', 'C':1.0, 'gamma':'scale', 'probability':True, 'class_weight':class_weights}
    svm_param_space = Space.from_yaml(hyperspaces_path, 'SVM').dimensions # get list of skopt.space dimenions

    ### BDT hyperparameters ###
    bdt_param = {'random_state':0, 'class_weight':class_weights}
    bdt_param_space = Space.from_yaml(hyperspaces_path, 'BDT').dimensions # get list of skopt.space dimenions

    ### BDF hyperparameters ###
    bdf_param = {'random_state':0, 'n_estimators':100, 'class_weight':class_weights}
    bdf_param_space = Space.from_yaml(hyperspaces_path, 'BDF').dimensions # get list of skopt.space dimenions

    ### GBC hyperparameters ###
    gbc_param = {'n_estimators':100, 'learning_rate':0.1}
    gbc_param_space = Space.from_yaml(hyperspaces_path, 'GBC').dimensions # get list of skopt.space dimenions

    ### LRC hyperparameters ###
    lrc_param = {'random_state':0, 'class_weight':class_weights}
    lrc_param_space = Space.from_yaml(hyperspaces_path, 'LRC').dimensions # get list of skopt.space dimenions

    ### XGB hyperparameters ###
    xgb_param = {'objective':'binary:logistic', 'use_label_encoder':False, 'eval_metric':'logloss'}
    xgb_param_space = Space.from_yaml(hyperspaces_path, 'XGB').dimensions # get list of skopt.space dimenions


    dict_models = {
        'NGB': (GaussianNB, None, range(len(metrics)), None),
        'SVM': (SVC, svm_param, range(len(metrics)), svm_param_space),
        'BDT': (DecisionTreeClassifier, bdt_param, range(len(metrics)), bdt_param_space),
        'XGB': (XGBClassifier, xgb_param, range(len(metrics)), xgb_param_space),
        'BDF': (RandomForestClassifier, bdf_param, range(len(metrics)), bdf_param_space),
        'GBC': (GradientBoostingClassifier, gbc_param, range(len(metrics)), gbc_param_space),
        'LRC': (LogisticRegression, lrc_param, range(len(metrics)), lrc_param_space),
        'SFS-BDF(2)': (RandomForestClassifier, bdf_param, sfs2.k_feature_idx_, bdf_param_space),
        'SFS-BDF(4)': (RandomForestClassifier, bdf_param, sfs4.k_feature_idx_, bdf_param_space)
    }

    runner = TrainModelsRunner(dict_models, df_sms, metrics, X, y)
    runner.fit()

    if runner.best_model['model_name'] != 'NGB':
        runner.optimize()

    #################################################
    # compute prediction proba on train and test set
    # these proba will be used in model DEPLOY stage, post-production analysis
    model = runner.best_model['clf']

    X_train = runner.X_train
    true_train_preds = model.predict_proba(X_train)[:, 1]

    X_test = runner.X_test
    true_test_preds = model.predict_proba(X_test)[:, 1]
    #################################################

    runner.store(location=backup_location)

    if make_plots:
        runner.plots()
    
    return runner.best_model, true_train_preds, true_test_preds

def model_stress_test(model, df_stress_dataset:pd.DataFrame, metrics:List[str]) -> None:
    '''
        Apply a model stress test to the best retrieved model.

        Parameters
        ----------
        model: object
            Best model to detect duplicate tickets

        df_stress_dataset: pd.DataFrame
            Dataframe for the stress test

        metrics: list
            List of metrics used to compute pair sentece similarity

        Returns
        -------
        None

    '''

    # MODEL STRESS TESTING
    X_test = np.array(df_stress_dataset.loc[:, metrics])
    y_test = np.array(df_stress_dataset.loc[:, 'duplication'])

    y_pred = model.predict(X_test)
    y_idx_failed = np.where(y_pred != y_test)[0] 

    df_stress_dataset.iloc[y_idx_failed, :].to_excel(backup_location+'stress_test.xlsx', 
                    sheet_name='Stress Test', index = False)
