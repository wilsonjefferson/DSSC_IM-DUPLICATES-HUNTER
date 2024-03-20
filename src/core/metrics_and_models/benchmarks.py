import numpy as np
import warnings
import logging
import Levenshtein as lev # pip install python-Levenshtein or conda install levenshtein
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from difflib import SequenceMatcher

from core.utils.Ticket import Ticket
from core.data_manipulation.term_counters import word_freq, doc_freq

log = logging.getLogger('benchmarksLogger')

vectorizer = CountVectorizer() # Convert a collection of text documents to a matrix of token counts.
lsa = TruncatedSVD(1, algorithm = 'arpack')

# TODO: Good candidate for cashing strategy
def metric_decorator(func):
    def wrapper(sentence1:object, sentence2:object, measure_name:str, ticket_config:str, **kwargs):
        if ticket_config is None:
            return func(sentence1, sentence1)
        
        sent1 = sentence1.config.get(ticket_config)
        sent2 = sentence2.config.get(ticket_config)
        log.debug(f'sentence1: {sentence1.raw} config: {ticket_config} sent1: {sent1}')
        log.debug(f'sentence2: {sentence2.raw} config: {ticket_config} sent2: {sent2}')

        if len(sent1)==0 or len(sent2)==0:
            log.warning(f'{measure_name}: configuration {ticket_config} provide empty sentence list.\n'
                        +'Sentences tokenization without stopwords are provided.')
            sent1 = sentence1.tokens_no_stop
            sent2 = sentence2.tokens_no_stop

        return func(sent1, sent2, **kwargs)
    return wrapper

@metric_decorator
def jaccard_similarity(tokens1:list, tokens2:list) -> float:
    '''
        The purpose of this method is to compute the Jaccard Similarity score
        between a pair of sentences.

        Parameters
        ----------
        tokens1: list
            Sentence/text list from ticket

        tokens2: list
            Sentence/text list from ticket

        Returns
        -------
            float
                Jaccard Similarity score
    '''
    
    tokens1 = set(tokens1)
    tokens2 = set(tokens2)

    if len(tokens1.union(tokens2)) == 0:
        log.debug('tokens1: %s\ntokens2: %s'%(tokens1, tokens2))
        return 0
    return len(tokens1.intersection(tokens2)) /len(tokens1.union(tokens2))

@metric_decorator
def dice_similarity(tokens1:list, tokens2:list) -> float:
    '''
        This method compute the DICE similarity between a pair of text.

        Parameters
        ----------
        sentence1: Ticket
            Sentence/text list from ticket

        sentence2: Ticket
            Sentence/text list from ticket

        Returns
        -------
            float
                DICE Similarity score
    '''

    tokens1 = set(tokens1)
    tokens2 = set(tokens2)
    
    return 2 * (len(tokens1 & tokens2)) / (len(tokens1) + len(tokens2))    

@metric_decorator
def lsa_similarity(tokens1:list, tokens2:list) -> float:
    '''
        This method compute the LSA similarity between a pair of text.

        Parameters
        ----------
        sentence1: list
            Sentence/text list from ticket

        sentence2: list
            Sentence/text list from ticket

        use_stopwords: bool
            If True, stopwords are used, False otherwise

        Returns
        -------
            float
                LSA Similarity score
    '''

    tokens1 = set(tokens1)
    tokens2 = set(tokens2)

    sent1 = ' '.join(tokens1)
    sent2 = ' '.join(tokens2)

    dtm = vectorizer.fit_transform([sent1, sent2]).asfptype()
    dtm_lsa = lsa.fit_transform(dtm)
    similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)

    return similarity[0, 1]

@metric_decorator
def lcs_similarity(sentence1:str, sentence2:str) -> float:
    '''
        This method compute the LCS similarity between a pair of text.

        Parameters
        ----------
        sentence1: Ticket
            Sentence/text from a ticket

        sentence2: Ticket
            Sentence/text from a ticket

        Returns
        -------
            float
                LCS Similarity score
    '''
    sent1 = sentence1
    sent2 = sentence2

    matcher = SequenceMatcher(None, sent1, sent2)  # matcher for longestSubstring (LCS)
    match = matcher.find_longest_match(0, len(sent1), 0, len(sent2))
    lcs = sent1[match.a: match.a + match.size] 
    return len(lcs) / len(max([sent1, sent2]))

@metric_decorator
def swo_idf_similarity(tokens1:list, tokens2:list, corpus:list, word_frequency:dict) -> float:
    '''
        The purpose of this method is to compute the Simple Word Overlapping Similarity 
        score between a pair of sentences.

        Parameters
        ----------
        corpus: list
            List of Ticket messages
        
        word_frequency: dict
            Dictionary of word frequency
        
        sentence1: Ticket
            Sentence/text from a ticket

        sentence2: Ticket
            Sentence/text from a ticket

        use_stopwords: bool
            If True, stopwords are used, False otherwise

        Returns
        -------
        sim: float
            Simple Word Overlapping Similarity score
    '''

    wf_1 = word_freq(tokens1)
    wf_2 = word_freq(tokens2)
    corpus_len = len(corpus)
    inter = set(tokens1).intersection(set(tokens2))

    sim = 1
    for token in inter:
        # sim += log(wf_1[token]+1)*log(wf_2[token]+1)*log((corpus_len+1) / (word_frequency[token]+0.5))

        token_in_wf_1 = token in wf_1.keys()
        token_in_wf_2 = token in wf_2.keys()
        token_in_word_frequency = token in word_frequency.keys()

        if not (token_in_wf_1 and token_in_wf_2 and token_in_word_frequency):
            log_message = f'token: {token} not a valid key. '
            log_message += f'{token} in wf_1: {token_in_wf_1} '
            log_message += f'{token} in wf_2: {token_in_wf_2} '
            log_message += f'{token} in word_frequency: {token_in_word_frequency}'
            log.warning(log_message)

            if not (token_in_wf_1 and token_in_wf_2):
                log.critical(f'{token} is not present in wf_1 or wf_2')
                raise KeyError
            elif not token_in_word_frequency:
                log.warning('tentative to adjust word frequency dictionary...')
                corpus_tokenized = (ticket.tokens for ticket in corpus)

                word_frequency = doc_freq(corpus_tokenized) # word document frequency
                if word_frequency.get(token, True) is None:
                    log.error(f'{token} is not a valid key in word_frequency: \n {token} added with value 0')
                    word_frequency[token] = 0

        sim *= (wf_1[token]+1) + (wf_2[token]+1)  + ((corpus_len+1) / (word_frequency[token]+0.5))
    sim = 1 - (1/sim)

    return sim

@metric_decorator
def pre_trained_average_word_embeddings_similarity(tokens1:list, tokens2:list, model) -> float:
    '''
        This method compute text similarity between texts using the word embeddings from
        a given pre-trained model. The Similarity score is computed as average of the words
        from each text and evaluating the similarity between these two average vectors.

        Parameters
        ----------
        sentence1: Ticket
            Sentence/text list from ticket

        sentence2: Ticket
            Sentence/text list from ticket

        model: object
            NLP Pre-trained model

        Returns
        -------
            float
                Word Embedding Similarity score
    '''

    tokens1 = [token for token in tokens1 if token in model]
    tokens2 = [token for token in tokens2 if token in model]

    m_tokens1 = [model[token] for token in tokens1]
    m_tokens2 = [model[token] for token in tokens2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        embedding1 = np.nanmean(m_tokens1, axis=0)
        embedding2 = np.nanmean(m_tokens2, axis=0)

    if embedding1.size == 0 or np.isnan(m_tokens1).all():
        log.debug('embedding1 is not valid - tokens1: %s' % ', '.join(map(str, tokens1)))
        return 0
    elif embedding2.size == 0 or np.isnan(m_tokens2).all():
        log.debug('embedding2 is not valid - tokens2: %s' % ', '.join(map(str, tokens2)))
        return 0

    return cosine_similarity([embedding1], [embedding2])[0][0]

@metric_decorator
def tf_idf_similarity(sentence1:str, sentence2:str) -> float:
    '''
        The purpose of this method is to compute the Cosine Similarty score  
        between a pair of sentences, by means of TF-IDF.

        Parameters
        ----------        
        sentence1: str
            Sentence/text from a ticket

        sentence2: str
            Sentence/text from a ticket

        Returns
        -------
            float
                Cosine Similarity score
    '''

    embedding1, embedding2 = sentence1, sentence2 # alias, just to clarify they are actually embeddings
    return cosine_similarity(embedding1, embedding2)[0][0]

@metric_decorator
def jw_similarity(tokens1:list, tokens2:list) -> float:
    '''
        The purpose of this method is to compute the Tree Edit Score Similarity 
        score between a pair of sentences.

        Parameters
        ----------      
        sentence1: Ticket
            Sentence/text list from ticket

        sentence2: Ticket
            Sentence/text list from ticket

        Returns
        -------
            float
                Tree Edit Score Similarity score
    '''

    return lev.jaro_winkler(tokens1, tokens2) # it may require strings instead of tokens

@metric_decorator
def bst_similarity(tokens1:list, tokens2:list, model) -> float: 
    '''
        The purpose of this method is to compute the Bert SentenceTransformer 
        Similarity score between a pair of sentences, nby means of Cosine Similarity.

        Parameters
        ----------      
        tokens1: list
            Sentence/text list from a ticket

        tokens2: list
            Sentence/text list from a ticket

        model: Transformer model
            Tipe of BERT model

        Returns
        -------
            float
                Cosine Similarity score
    '''
    sentence1 = ' '.join(tokens1)
    sentence2 = ' '.join(tokens2)
    
    sentence_embeddings = model.encode([sentence1, sentence2], show_progress_bar=False) # verbose=None
    return cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])[0][0]

@metric_decorator
def fasttext_similarity(tokens1:str, tokens2:str, model) -> float:
    '''
        The purpose of this method is to compute the FastText Similarity
        score between a pair of sentences, nby means of Cosine Similarity.

        Parameters
        ----------
        sentence1: str
            Sentence/text from a ticket

        sentence2: str
            Sentence/text from a ticket

        model: FastText model
            Pretrained FastText on train/test dataset

        Returns
        -------
            float
                Cosine Similarity score
    '''
    
    sentence1 = ' '.join(tokens1)
    sentence2 = ' '.join(tokens2)
    sentence_embedding1 = model.get_sentence_vector(sentence1)
    sentence_embedding2 = model.get_sentence_vector(sentence2)
    return cosine_similarity([sentence_embedding1], [sentence_embedding2])[0][0]

@metric_decorator
def doc2vec_similarity(tokens1:list, tokens2:list, model) -> float:
    '''
        The purpose of this method is to compute the Doc2Vec Similarity score 
        between a pair of sentences, nby means of Cosine Similarity.

        Parameters
        ----------      
        tokens1: list
            Sentence/text list from a ticket

        tokens2: list
            Sentence/text list from a ticket

        model: Gensim model
            Gensim Doc2Vec model

        Returns
        -------
            float
                Cosine Similarity score
    '''
    sentence_embedding1 = model.infer_vector(tokens1)
    sentence_embedding2 = model.infer_vector(tokens2)
    return cosine_similarity([sentence_embedding1], [sentence_embedding2])[0][0]
