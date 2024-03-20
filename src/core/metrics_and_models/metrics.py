import os
import pickle
import joblib
import numpy as np
import pandas as pd
import functools as ft
from copy import deepcopy
from itertools import product
from typing import Callable, Dict

import logging
log = logging.getLogger('metricsLogger')

import core.metrics_and_models.benchmarks as metrics
from core.data_manipulation.term_counters import build_corpus, doc_freq

from ConfigNameSpace import package_root, MAIN_STAGE, MAIN_TRAIN
train_backup_location = MAIN_TRAIN.backup_location
backup_location = MAIN_STAGE.backup_location
backup_location_plots = MAIN_STAGE.backup_location + '/plots/'
make_plots = MAIN_STAGE.make_plots

## PRE-TRAINED MODELS ##
import gensim
PATH_TO_WORD2VEC = package_root+'gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC, binary=True)

# PATH_TO_DOC2VEC = train_backup_location+'doc2vec/doc2vec_pretrained_model.model'
# doc2vec = gensim.models.doc2vec.Doc2Vec.load(PATH_TO_DOC2VEC)

PATH_TO_GLOVE = package_root+'gensim-data/glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz'
glove2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_GLOVE)

import fasttext
fasttext = fasttext.load_model(train_backup_location+'fasttext/fasttext_pretrained_model.bin')

from sentence_transformers import SentenceTransformer
bert = SentenceTransformer(package_root+'huggingface/sentence-transformers/all-mpnet-base-v2')
########################

## PRE-TRAINED VECTORIZERS ##
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = joblib.load(train_backup_location + 'tfidf/tfidf_pretrained_vectorizer.joblib')
#############################

from sklearn.preprocessing import MinMaxScaler
normalizer = MinMaxScaler() # Min Max normalizer


def run_similarity_metrics(df:pd.DataFrame, dict_benchmarks:Dict[str, Callable]) -> dict:
    '''
        The purpose of this method is to create a dictionary wit hthe similarity
        scores, for each pair of tickets, by means of a list of similarity measures.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame of the tickets

        benchmarks: dict
            Dictionary of similarity methods to evaluate the similarity between pairs
            of tickets
        
        Returns
        -------
        sims: dict
            Dictionary of similarity scores
    '''

    location = backup_location + 'step_compute_similarities/'
    if os.path.exists(location+'sims_results.pickle'):
        sims_results = pd.read_pickle(location+'sims_results.pickle')
        return sims_results

    sims = {label:[] for label in dict_benchmarks.keys()}
    sims['origin_incidentid'] = df.origin_incidentid
    sims['duplicate_incidentid'] = df.duplicate_incidentid
    sims['duplication'] = df.duplication

    sentences = list(zip(df['origin_ticket_cleaned'].tolist(), df['duplicate_ticket_cleaned'].tolist()))
    tf_idf = {'TF-IDF':dict_benchmarks.pop('TF-IDF')}

    log.info('create product all object...')
    prd_all = product(sentences, dict_benchmarks)

    log.info('create tf-idf product object...')
    prd_tf_idf = product(sentences, tf_idf)

    for sentences, label in prd_tf_idf:
        method = tf_idf[label]
        try:
            tfidf_matrix = tfidf.transform([sentences[0].raw, sentences[1].raw])
            sentences_embeddings = (tfidf_matrix[0], tfidf_matrix[1])
        except Exception as e:
            log.exception(f'tfidf transformation sentence1: {sentences[0].raw}')
            log.exception(f'tfidf transformation sentence2: {sentences[1].raw}')
            log.exception(e)
            sentences_embeddings = (None, None)

        sim = 0.0 if sentences_embeddings[0] is None or sentences_embeddings[1] is None else method(*sentences_embeddings)
        sims[label].append(sim)

    for sentences, label in prd_all:
        method = dict_benchmarks[label]
        sim = method(*sentences)
        sims[label].append(sim)

    sims_results = pd.DataFrame(sims)
    sims_results.to_pickle(location+'sims_results.pickle')

    return sims_results

def compute_metrics(df:pd.DataFrame, destination:str = None, custom_name_png:str = '') -> tuple:
    '''
        This method compute the similarity metrics for pair of tickets.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame of the dataset to use to train the model(s)

        destination: str
            It is the location where dataset + measures will be saved

        build_catalog: bool
            Parameter to specify if the catalog is under construction or not

        Returns
        -------
        metrics: list
            List with the name of the metrics used to evaluate the similarities

        sims_results_min_max_scaled: pd.DataFrame
            Dataframe with the similarity scores for each pair of sentence
    '''
    
    log.info('build corpus...')
    df, corpus = build_corpus(df) # construct corpus from df dataframe

    log.info('build lemmatized corpus...')
    corpus_tokenized = list(ticket.lemma for ticket in corpus)
    
    log.info('build word document frequency...')
    word_df = doc_freq(corpus_tokenized) # word document frequency

    dict_benchmarks = {
        "JCD": ft.partial(metrics.jaccard_similarity, measure_name='JCD', ticket_config='lemma'),
        "DICE": ft.partial(metrics.dice_similarity, measure_name='DICE', ticket_config='lemma'),
        # "LSA": ft.partial(lsa_similarity, measure_name='LSA', ticket_config='lemma'),
        "LCS": ft.partial(metrics.lcs_similarity, measure_name='LCS"', ticket_config='raw'),
        "SWO": ft.partial(metrics.swo_idf_similarity, measure_name='SWO', ticket_config='lemma', corpus=corpus, word_frequency=word_df),
        "TF-IDF": ft.partial(metrics.tf_idf_similarity, measure_name='TF-IDF', ticket_config=None), # ok
        "GLOVE": ft.partial(metrics.pre_trained_average_word_embeddings_similarity, measure_name='GLOVE', ticket_config='tokens_no_stop', model=glove2vec),
        # "DOC2VEC": ft.partial(metrics.doc2vec_similarity, measure_name='DOC2VEC',ticket_config='tokens_no_stop', model=doc2vec),
        "W2VEC": ft.partial(metrics.pre_trained_average_word_embeddings_similarity, measure_name='W2VEC', ticket_config='tokens_no_stop', model=word2vec),
        "WO": ft.partial(metrics.jaccard_similarity, measure_name='WO', ticket_config='n_grams_no_stop'),
        "JW": ft.partial(metrics.jw_similarity, measure_name='JW', ticket_config='lemma'),
        "BST": ft.partial(metrics.bst_similarity, measure_name='BST', ticket_config='tokens_no_stop', model=bert), # comment it for SOFT_0_duplicates_model
        "FASTTEXT": ft.partial(metrics.fasttext_similarity, measure_name='FASTTEXT', ticket_config='tokens_no_stop', model=fasttext)
    }

    list_metrics = list(dict_benchmarks.keys())

    log.info('Run similarities...')
    sims_results = run_similarity_metrics(df, dict_benchmarks)

    ### normalization ###
    log.debug('Apply normalization')
    sims_results_min_max_scaled = sims_results.copy()
    if len(sims_results_min_max_scaled) > 1:
        # apply normalization techniques by Column 1
        # it is note possible to normalize if you have one single row...
        not_normalize = ['JCD', 'DICE', 'LCS', 'SWO', 'TF-IDF', 'WO', 'JW', 'origin_incidentid', 'duplicate_incidentid', 'duplication']
        for column in sims_results.columns.difference(not_normalize):
            data = np.array([sims_results_min_max_scaled[column]]).reshape(-1, 1)
            normalized = normalizer.fit_transform(data).reshape(1, -1).tolist()[0]
            sims_results_min_max_scaled[column] = normalized
    ######################
    sims_results_min_max_scaled['duplication'] = sims_results_min_max_scaled['duplication'].astype(object)

    if make_plots:
        from core.utils.plots.metrics_plots import plot_metrics_distr

        dt = pd.merge(df, sims_results_min_max_scaled, 
                on=['origin_incidentid', 'duplicate_incidentid', 'duplication'], 
                how='inner', validate="one_to_one")

        if destination:
            dt.to_excel(destination, index=False)

        dt_plot_metrics = dt[['duplication'] + list_metrics]

        plot_metrics_distr(dt_plot_metrics, False, custom_name_png)
        plot_metrics_distr(dt_plot_metrics, True, custom_name_png + 'categorized_')

    return list_metrics, sims_results_min_max_scaled
