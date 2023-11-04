import pandas as pd
import matplotlib.pyplot as plt

from ConfigNameSpace import MAIN_STAGE
backup_location_plots = MAIN_STAGE.backup_location + '/plots/'

savefig = lambda name: plt.savefig(name, bbox_inches='tight')

def plot_metrics_distr(df:pd.DataFrame, groupby_target:bool, custom_name_png:str = '') -> None:
    
    title = 'Normalized Metrics Distribution'
    xlabel = 'Metrics Score'
    ylabel = 'Frequency'
    
    if groupby_target:

        # Create a figure with two subplots (1 row, 2 columns)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 4))

        classes = ['Negative', 'Positive']

        # Create histograms for each column and plot them in the subplots
        for i, c in enumerate(classes):
            ax = axes[i]
            ax.hist(df[df['duplication'] == i], label=f'class_{i}')
            ax.set_title(f'{c} category')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        # Adjust spacing between subplots
        plt.tight_layout()
    else:
        df = df.round(2)
        df = df.drop(columns=['duplication'])
        plt.figure(figsize=(15, 10))
        plt.hist(df.values, label=df.columns)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
    
    savefig(backup_location_plots+f'{custom_name_png}metrics_distribution.png')
    plt.close()
