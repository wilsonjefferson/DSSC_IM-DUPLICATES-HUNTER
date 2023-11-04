import pickle
import pandas as pd

import src
import ConfigNameSpace

ConfigNameSpace.MAIN_STAGE = ConfigNameSpace.MAIN_DEPLOY

backup_location = ConfigNameSpace.MAIN_STAGE.backup_location
model_path = ConfigNameSpace.MAIN_TRAIN.backup_location + 'dict_best_model.pickle'

import core


from model_deploy_components import (
    import_data, 
    filter_invalid_tickets, 
    reduce_search_space,
    text_preprocessing, 
    assess_on_new_tickets, 
    update_catalog, 
    store_recognize_duplicates,
    post_production_analysis
)


if __name__ == '__main__':

    df_target = import_data(query='RETRIEVE_YESTERDAY')
    
    df_target = filter_invalid_tickets(df_target)
    df_target, df_2_weeks_origins, catalog = reduce_search_space(df_target)
    df = text_preprocessing(df_target)
    df = df.drop_duplicates(subset=['incidentid'])

    # load ad-hoc pre-trained model during testing
    with open(model_path, 'rb') as f:
        dict_model = pickle.load(f)
    
    catalog, dict_recognized_duplicates, _ = assess_on_new_tickets(dict_model, df, 
                                                                   catalog, 
                                                                   df_2_weeks_origins)
    
    store_recognize_duplicates(df, catalog, dict_recognized_duplicates)
    df_catalog = update_catalog(catalog, df_2_weeks_origins)

    # STORE CATALOG (I.E. DICTIONARY OF ORIGINS AND STARS)
    with open(backup_location + 'catalog.pickle', 'wb') as handle:
        pickle.dump(catalog, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pd.ExcelWriter(backup_location + 'catalog.xlsx', engine='openpyxl') as writer:
        df_catalog.to_excel(writer, index=False)

    post_production_analysis()
