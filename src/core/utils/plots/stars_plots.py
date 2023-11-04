"""
    This script contains some methods to categorize stars
    according their depth (longest chain of duplicate tickets in a star).

    @file: im_stars_representation.py
    @version: v. 1.2.1
    @last update: 23/06/2022
    @author: Pietro Morichetti
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import random

from core.data_manipulation.itsm_tickets_constrains import check_violation_cycles_in_star
from ConfigNameSpace import MAIN_STAGE

backup_location_plots = MAIN_STAGE.backup_location + '/plots/'


def stars_per_depth(duplicates_dict:dict) -> dict:
    '''
        The purpose of this method is to create a dictionary where
        the stars origin tickets are the keys and as value the length
        of the deepest chain in the star.
        In this way we can easily categorize stars according the "depth".

        Parameters
        ----------
        duplicates_dict: dict
            Dictionary of: ticket -> origin ticket (or star)

        Returns
        -------
        stars_depth_dict: dict
            Dictionary of: origin ticket -> depth of the star of the origin ticket
    '''

    stars_depth_dict = {}
    for origin, duplicates in duplicates_dict.items():
        if type(duplicates) is nx.DiGraph:     
            # check violation: look for cycles in the star
            check_violation_cycles_in_star(duplicates)
            # longest path in the entire DAG: in the star cofiguration
            # it results in the longest path from the origin to the farest duplicate
            stars_depth_dict[origin] = nx.dag_longest_path_length(duplicates)
    return stars_depth_dict

def stars_per_category(duplicates_dict:dict, stars_origins:dict) -> dict:
    '''
        this method group stars according the depth category.
        See im_stars_representation.stars_per_depth method fpr 
        further details.

        Parameters
        ----------
        duplicates_dict: dict
            Dictionary of: ticket -> origin ticket (or star)

        stars_origins: dict
            Dictionary of: origin ticket -> depth of the star of the origin ticket

        Returns
        -------
        stars_per_category: dict
            Dictionary[star depth][origin ticket] = star of the origin ticket
    '''

    stars_per_category = {}
    for origin, path_length in stars_origins.items():
        key = 'Path Length: {}'.format(path_length)
        if not stars_origins.get(key):
            stars_per_category[key] = {}
        stars_per_category[key][origin] = duplicates_dict[origin]
    return stars_per_category

def plot_representative_stars_per_category(stars_per_category:dict, custom_name_png:str = '') -> None:
    '''
        Plot a representative star per depth category, the representative
        star(s) has the major number of nodes (duplicate tickets), so they 
        are the biggest stars per their own depth category.

        Parameters
        ----------
        stars_per_category: dict
            Dictionary[star depth][origin ticket] = star of the origin ticket

        Returns
        -------
        None
    '''

    for category, stars_category_dict in stars_per_category.items():
        n_duplicates = 0
        candidate_origin = None
        candidate_star = None

        print(f'category: {category}')

        for origin, star in stars_category_dict.items():
            print(origin, star, len(star))
            if n_duplicates < len(star):
                n_duplicates = len(star)
                candidate_origin = origin
                candidate_star = star
        
        if candidate_star is None:
            print('candidate star is None!')
            continue

        colors = ['orange' if node == candidate_origin else 'blue' for node in candidate_star.nodes]
        fig = plt.figure()
        nx.draw_networkx(stars_category_dict[candidate_origin], node_color=colors, with_labels=True)
        path_length = [int(s) for s in category.split() if s.isdigit()][0]
        fig.savefig(backup_location_plots+'{}path_length_{}.png'.format(custom_name_png, path_length))

def plot_scatter_stars(stars_dict:dict, custom_name_png:str = '') -> None:
    '''
        Plot how stars are distributed according the number of "neighbours" for 
        the origin of a star (i.e. its the duplicates), and the number of stars
        with the specified number of "neighbours"; x axis and y axis respectively.

        Parameters
        ----------
        stars_dict: dict
            Dictionary of: ticket origin -> star

        Returns
        -------
        None
    '''
    
    dict_n_nodes_per_n_stars = {}
    for star in stars_dict.values():
        n_nodes = star.number_of_nodes() - 1 # exclude the origin from the count
        if n_nodes not in dict_n_nodes_per_n_stars.keys():
            dict_n_nodes_per_n_stars[n_nodes] = 0
        dict_n_nodes_per_n_stars[n_nodes] += 1

    plt.scatter(dict_n_nodes_per_n_stars.keys(), dict_n_nodes_per_n_stars.values())
    plt.xlabel('N. node neighbours to the origin(s)')
    plt.ylabel('N. stars per number of neighbours')
    plt.title('Origins Neighbours Scatter Plot')
    plt.grid(False)
    plt.savefig(backup_location_plots+'{}n_nodes_per_n_stars.png'.format(custom_name_png), 
        bbox_inches='tight')

def plot_duplicates_dates_frequency(date_lists:list) -> None:
    """
    Plot the frequency of dates for each year within a specified range for each sublist.

    This function generates a scatter plot to visualize the frequency of dates for each year within the range.
    Each sublist is assigned a unique color, and the x-axis represents the years. 
    The plot is saved as 'duplicates_dates_frequency.png'.

    Parameters
    ----------
    date_lists: List[List[Union[str, datetime]]]
        A list of lists where each sublist contains date strings or datetime objects.
    """
    from collections import Counter
    from datetime import datetime

    all_years = [date.strftime("%Y") for sublist in date_lists for date in sublist]
    year_counts = Counter(all_years)

    # Plotting
    plt.figure(figsize=(10, 6))
    years, year_freqs = zip(*year_counts.items())
    plt.bar(years, year_freqs)

    # Set x-axis ticks to represent years
    # min_year, max_year = min(year_counts.keys()), max(year_counts.keys())
    # plt.xticks(range(min_year, max_year))

    plt.title('Frequency of Dates for Each Year')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.savefig(backup_location_plots+'duplicates_dates_frequency.png')
