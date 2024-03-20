import pandas as pd
import pickle

import src
import ConfigNameSpace

ConfigNameSpace.MAIN_STAGE = ConfigNameSpace.MAIN_BUILD

backup_location = ConfigNameSpace.MAIN_STAGE.backup_location
model_path = ConfigNameSpace.MAIN_TRAIN.backup_location + 'dict_best_model.pickle'

import core

from model_deploy_components import (
    import_data, 
    filter_invalid_tickets,
    reduce_search_space, 
    text_preprocessing, 
    assess_on_new_tickets, 
    update_catalog
)


if __name__ == '__main__':

    df_two_weeks, catalog, df_tmp = None, None, None

    df_two_weeks  = import_data(query='RETRIEVE_LAST_2_WEEKS')

    df_two_weeks  = filter_invalid_tickets(df_two_weeks)
    _, _, catalog = reduce_search_space(df_two_weeks)
    df_two_weeks = text_preprocessing(df_two_weeks)
    df_two_weeks = df_two_weeks.drop_duplicates(subset=['incidentid'])

    # load ad-hoc pre-trained model during testing
    with open(model_path, 'rb') as f:
        dict_model = pickle.load(f)
    
    catalog, _, _ = assess_on_new_tickets(dict_model, df_two_weeks, catalog, _)

    # this step is "useless" in this phase of the project because
    # the catalog is just now created, so no need to update it,
    # what is doing this function, at this stage, is just to provide
    # a dataframe shape to the catalog.
    df_catalog = update_catalog(catalog, df_two_weeks)

    # STORE CATALOG (I.E. DICTIONARY OF ORIGINS AND STARS)
    with open(backup_location + 'catalog.pickle', 'wb') as handle:
        pickle.dump(catalog, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pd.ExcelWriter(backup_location + 'catalog.xlsx', engine='openpyxl') as writer:
        df_catalog.to_excel(writer, index=False)
