import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer
from nltk.util import ngrams

import gensim
import spacy

from ConfigNameSpace import package_root


stopwords_dir = package_root+'nltk_data/corpora/stopwords'
punkt_dir = package_root+'nltk_data/tokenizers/punkt'
wordnet_dir = package_root+'nltk_data/corpora/wordnet'
omw_dir = package_root+'nltk_data/corpora/omw-1.4'
spacy_dir = package_root+'spacy-data/en_core_web_lg-3.4.0/en_core_web_lg/en_core_web_lg-3.4.0'

# NOTE: need to manually download StopwWords from http://www.nltk.org/nltk_data/
nltk.download('stopwords', download_dir=stopwords_dir)
# NOTE: need to manually download Punkt Tokenizer Models from http://www.nltk.org/nltk_data/
nltk.download('punkt', download_dir=punkt_dir) 
# NOTE: need to manually download Wordnet 3.1 from http://www.nltk.org/nltk_data/
nltk.download('wordnet', download_dir=wordnet_dir)
# NOTE: need to manually download Open Multilingual Wordnet from http://www.nltk.org/nltk_data/
nltk.download('omw-1.4', download_dir=omw_dir)

nlp = spacy.load(spacy_dir)
nlp.Defaults.stop_words |= {'null'} # include "null" term in the stop word collection

STOPWORDS = set(nltk.corpus.stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
mwe_tokenizer = MWETokenizer()


### Read each Excel file into a separate DataFrame ###
import glob
import pandas as pd
root_sources = 'sources/text_preprocessing/'
folder_path = root_sources
xls_files = glob.glob(f"{folder_path}*.xls")

dfs = [pd.read_excel(file_path) for file_path in xls_files]
df_ecb_acronyms = pd.concat(dfs, axis=0, ignore_index=True, names=['Stands_for'])
list_ecb_acronyms = df_ecb_acronyms['Stands_for'].tolist()
list_ecb_acronyms = [string.split() for string in list_ecb_acronyms]

mwe_tokenizer.add_mwe(['[', 'PERSON', ']']) # special token
mwe_tokenizer.add_mwe(['[', 'HTTP', ']']) # special token
mwe_tokenizer.add_mwe(['[', 'DATETIME', ']']) # special token
mwe_tokenizer.add_mwe(['[', 'DATE', ']']) # special token

for topical_words in list_ecb_acronyms:
    mwe_tokenizer.add_mwe(topical_words)
######################################################


class Ticket:
    '''
        This class is an abstraction of ticket in ITSM and it automatically performs
        a collection of pre-processing functions to the ticket itself.

        Attributes
        ----------
        raw: str
            The original raw ticket from DISC

        tokens: list
            The ticket splitted in tokens

        tokens_no_stop: list
            Tokenized ticket without stop words
        
        n_grams_no_stop: list
            Tokenized ticket without stop words and grouped in n grams

        lemma: list
            Tokenized ticket without stop words and each token lemmatized

        config: dict
            Dictionary as a collection of all the other attributes
    '''
    
    def __init__(self, ticket:str):
        self.raw = ticket
        self.tokens = self.tokenize(ticket)
        self.tokens_no_stop = self.no_stop_words(self.tokens)
        self.n_grams_no_stop = self.n_grams(self.tokens_no_stop)
        self.lemma = self.lemmatize(' '.join(self.tokens_no_stop))
        self.config = {
            'raw': self.raw,
            'tokens': self.tokens,
            'tokens_no_stop': self.tokens_no_stop,
            'n_grams_no_stop': self.n_grams_no_stop,
            'lemma': self.lemma
        }

    def tokenize(self, ticket:str) -> list:
        '''
            Tokenize string in tokens.

            Parameters
            ----------
            ticket: str
                row string

            Returns
            -------
                list
                    list of tokens
        '''

        tokens = word_tokenize(ticket)

        # Tokenize the text using the MWE tokenizer
        tokens_with_topical_words = mwe_tokenizer.tokenize(tokens)
        return tokens_with_topical_words

    def no_stop_words(self, ticket:list) -> list:
        '''
            Remove stopwords tokens.

            Parameters
            ----------
            ticket: list
                List of tokens

            Returns
            -------
                list
                    filtered list of tokens
        '''

        return [token for token in ticket if token not in STOPWORDS]

    def _bi_tri_grams(self, ticket: str, min_count: int = 3, threshold: int = 100) -> tuple:
        '''
            Generate set of bi-grams and tri-grams.

            Parameters
            ----------
            ticket: list
                List of tokens

            min_count: int
                Ignore words with length less or equal to min_count

            threshold: int
                Score threshold to format Phareses object

            Returns
            -------
                tuple
                    bi-grams and tri-grams
        '''

        # Build the bigram and trigram models

        # higher threshold fewer phrases.
        bigram = gensim.models.Phrases(ticket, min_count=min_count, threshold=threshold)
        trigram = gensim.models.Phrases(bigram[ticket], threshold=threshold)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        return bigram_mod, trigram_mod

    def n_grams(self, ticket:list, n_words:int = 2) -> list:
        '''
            Generate n_grams.

            Parameters
            ----------
            ticket: list
                List of tokens

            n_words: int
                Number of words that will form the gram

            Returns
            -------
                list
                    List of n-grams
        '''

        return list(ngrams(ticket, n_words))

    def lemmatize(self, ticket:str) -> list:
        '''
            Lemmatization on no-stopwords tokens.

            Parameters
            ----------
            ticket: list
                List of tokens

            Returns
            -------
                list
                    List of lemmatized token
        '''

        lemma_ticket = [lemmatizer.lemmatize(token.text) 
                for token in nlp(ticket) 
                if len(token) > 2 and token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}]
        return lemma_ticket
