import pandas as pd


if __name__ == '__main__':

    data_measure1 = r'D:\2022-10-24_SOFT_train_data_measure.xlsx'
    data_measure2 = r'D:\2022-10-25_SOFT_train_data_measure.xlsx'
    destination = r'D:\2022-10-26_SOFT_train_data_measure.xlsx'

    df_main = pd.read_excel(data_measure1)
    df_m1 = pd.read_excel(data_measure2)

    df_m1 = df_m1[['origin_incidentid', 'duplicate_incidentid', 'SWO']]

    df = df_main.merge(df_m1, on=['origin_incidentid', 'duplicate_incidentid'])
    df.to_excel(destination, index=False)
