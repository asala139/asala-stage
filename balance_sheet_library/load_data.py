import os
import pandas as pd

def load_data_by_years(start_year, end_year, dir="by_years/", numero_anni_disponibili=10):
    df_global = []
    for year in range(start_year, end_year+1):
        f = str(year)+'.csv'
        df_years = pd.read_csv(dir + f, index_col=False)
        df_years = df_years[df_years['Numero di anni disponibili'] == numero_anni_disponibili]
        if len(df_global) == 0:
            df_global = df_years
        else:
            df_global = pd.concat([df_global, df_years])
    df_global.drop('Unnamed: 0', inplace=True, axis=1)
    df_global.reset_index(drop=True, inplace=True)
    return df_global

def load_data_by_years_orbis(start_year, end_year, dir="by_years/"):
    df_global = []
    for year in range(start_year, end_year+1):
        f = str(year)+'.csv'
        df_years = pd.read_csv(dir + f, index_col=False)
        df_years['years'] = [year] * len(df_years)
        new_columns = []
        for c in df_years.columns:
            new_c = ''.join([i for i in c if not i.isdigit()])
            new_columns.append(new_c)
        df_years.columns = new_columns
        #df_years = df_years[df_years['Numero di anni disponibili'] == numero_anni_disponibili]
        if len(df_global) == 0:
            df_global = df_years
        else:
            df_global = pd.concat([df_global, df_years], axis=0)
    #df_global.drop('Unnamed: 0', inplace=True, axis=1)
    df_global.reset_index(drop=True, inplace=True)
    return df_global
