import pandas as pd
from collections import Counter
import spacy

from ConfigNameSpace import package_root
spacy_dir = package_root+'spacy-data/en_core_web_lg-3.4.0/en_core_web_lg/en_core_web_lg-3.4.0'

nlp = spacy.load(spacy_dir)
nlp.Defaults.stop_words |= {'null'} # include "null" term in the stop word collection

from core.utils.Ticket import Ticket

from core.utils.plots.text_plots import(
    plot_nlp_frequency,
    plot_word_density,
    plot_wordcloud
)

N_RECORDS = 5000

flat = lambda ll: [item for l in ll for item in l] # flat list


def text_analysis(df:pd.DataFrame):
    
    text_cleaned_tokenized = df.apply(lambda x: Ticket(x).tokens).iloc[:N_RECORDS, :]
    text_cleaned_lemma = df.apply(lambda x: Ticket(x).lemma).iloc[:N_RECORDS, :]

    # ## POS TAGS
    # # collect pos_tags
    pos_tags = [[(token.text, token.pos_) for token in nlp(sentence)] for sentence in text_cleaned_tokenized]
    _ = nlp_frequency('POS-TAGS', pos_tags) # return df_pos_tags

    # ## NERS
    # # collect NER from cleaned text
    ners = [[(token.text, token.label_) for token in nlp(sentence).ents] for sentence in text_cleaned_tokenized] 
    _ = nlp_frequency('NER', ners) # return df_ners

    ## Word Frequency
    # flat list in a unique large tokens list
    text_tokenized_merged = flat(text_cleaned_tokenized)
    dict_text_token_freq = Counter(text_tokenized_merged)

    df_text_words_freq = pd.DataFrame(dict_text_token_freq.most_common(30), columns=['Words', 'Frequency'])

    ## Plots
    plot_nlp_frequency(df_text_words_freq)
    plot_word_density(pd.DataFrame.from_dict({'text_cleaned':text_cleaned_lemma}))
    plot_wordcloud(dict_text_token_freq, 'Cleaned Description')

def nlp_frequency(analysis:str, l:list) -> pd.DataFrame:
    '''
      The purpose of this method is to preapare the NLP frequency dictionary
      and call the plot_nlp_frequency.

      Parameters
      ----------
      analysis: str
          Type of NLP analysis performed

      l: list
          List of NLP items (according the type of analysis performed)

      Returns
      -------
      df: pd.DataFrame
          Dataframe of NLP object frequency
          
    '''

    # flat list in a unique large list
    l_merged = flat(l)

    # filter: consider unique items
    l_merged = list(set(l_merged))
    l_merged = Counter([pos[1] for pos in l_merged]).most_common() # print most common

    df = pd.DataFrame(l_merged, columns=[analysis, 'Frequency'])
    plot_nlp_frequency(df)
    return df