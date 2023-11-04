import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

from ConfigNameSpace import MAIN_STAGE

backup_location_plots = MAIN_STAGE.backup_location + '/plots/'
savefig = lambda name: plt.savefig(name, bbox_inches='tight')

def plot_wordcloud(dict_freq:dict, column:str) -> None:
    '''
        The purpose of this method is to plot a Wordcloud.

        Parameters
        ----------
        dict_freq: dict
            Dictionary of word frequency

        column: str
            WordCloud is referring to a certain column name

        Returns
        -------

    '''

    wordcloud = WordCloud(width=1000, height=500, stopwords=STOPWORDS, 
        max_words=100, background_color='white').generate_from_frequencies(dict_freq)
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title('%s Wordcloud' % column)
    savefig(backup_location_plots+'{}_wordcloud.png'.format(column))
    plt.show()

def plot_word_density(df:pd.DataFrame) -> None:
    '''
        The purpose of this method is to plot the words density Histogram.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe of tickets

        Returns
        -------
        
    '''

    column = df.columns[0]
    df['word_count'] = df[column].apply(lambda x : len(x.split()))
    df['char_count'] = df[column].apply(lambda x : len(x.replace(" ","")))
    df['word_density'] = df['word_count'] / (df['char_count'] + 1)
    
    plt.hist(df['word_density'])
    plt.xlabel('Word density')
    plt.ylabel('Number of tickets')
    plt.title('Word density Histogram')
    plt.grid(False)
    savefig(backup_location_plots+'wordensity.png')
    plt.show()

def plot_nlp_frequency(df:pd.DataFrame) -> None:
    '''
        The purpose of this method is to plot the frequency of certain NLP
        objects (ex. POS-TAGs, NERs, ...), by means of a barplot.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe of NLP object frequency

        Returns
        -------
        
    '''

    analysis = df.columns[0]
    plt.bar(df[analysis], df[df.columns[1]])
    plt.xticks(rotation = 60)
    plt.ylabel('Frequency')
    plt.xlabel(analysis)
    plt.title(analysis + ' Frequency Trend')
    savefig(backup_location_plots+'{}.png'.format(analysis))
    plt.show()
