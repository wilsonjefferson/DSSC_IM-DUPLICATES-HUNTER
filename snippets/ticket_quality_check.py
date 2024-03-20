import pandas as pd


if __name__ == '__main__':

    source = r'\\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\multiple2.xlsx'
    destination = r'\\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\random_sample.xlsx'
    destination2 = r'\\GIMECB01\HOMEDIR-MP$\morichp\IT Services Management\Incident Management\python-projects\filtering-of-recognisable-duplicate-tickets\sources\random_sample.xlsx'

    source = pd.ExcelFile(source)
    df = pd.read_excel(source, dtype=str, sheet_name='case 1 2 4')
    df = pd.concat([df, pd.read_excel(source, dtype=str, sheet_name='case 3')], axis=0)
    df = pd.concat([df, pd.read_excel(source, dtype=str, sheet_name='case 5')], axis=0)

    df.sort_values("incidentid", inplace = True)
    df.drop_duplicates(subset ="incidentid", keep = False, inplace = True)
    

    df2 = pd.read_excel(destination, dtype=str)
    df2.reset_index()
    
    for idx in df.index:
        if str(df.at[idx, 'incidentid']) in list(df2['incidentid']):
            df.drop([idx], inplace=True)
    
    df_sample = df.sample(n=1100)
    df_sample.to_excel(destination2)
    