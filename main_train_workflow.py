import src
import ConfigNameSpace

ConfigNameSpace.MAIN_STAGE = ConfigNameSpace.MAIN_TRAIN

import core # NOTE: Do not delete, required to trigger the core.__init__


from model_train_components import (
    import_data, 
    filter_invalid_tickets, 
    construct_duplicates_dict, 
    text_preprocessing,
    pretrain_tfidf,
    pretrain_fasttext,
    pretrain_doc2vec,
    text_descriptive_analysis,
    construct_dataset_original, 
    compute_similarities,
    explore_models,
    # construct_dataset_artificial,
    # model_stress_test
)


if __name__ == '__main__':

    df_original, df, dict_stars_filtered, df_dataset  = None, None, None, None
    df_original = import_data()
    print('0 workflow checking df_original - %s %s'%(len(df_original.incidentid), len(df_original.incidentid.unique())))

    df = filter_invalid_tickets(df_original)
    print('1 workflow checking df - %s %s'%(len(df.incidentid), len(df.incidentid.unique())))

    df, dict_stars_filtered = construct_duplicates_dict(df)
    print('2 workflow checking df - %s %s'%(len(df.incidentid), len(df.incidentid.unique())))
    print('2 workflow checking dict_stars_filtered - %s'%len(dict_stars_filtered))

    df, dict_stars_filtered, dict_origin_duplicates = text_preprocessing(df, dict_stars_filtered)
    print('3 workflow checking df - %s %s'%(len(df.incidentid), len(df.incidentid.unique())))
    print('3 workflow checking dict_stars_filtered - %s'%len(dict_stars_filtered))
    df = df.drop_duplicates(subset=['incidentid'])

    _ = pretrain_tfidf(df)
    _ = pretrain_fasttext(df, dict_origin_duplicates)
    # _ = pretrain_doc2vec(df)
    text_descriptive_analysis(df['text_cleaned'])

    df_dataset = construct_dataset_original(df, dict_stars_filtered)
    df_sms, metrics = compute_similarities(df_dataset)

    model = explore_models(df_sms, metrics)

    # df_stress_test = construct_dataset_artificial(df, dict_stars_filtered)
    # model_stress_test(model, df_stress_test, metrics)
