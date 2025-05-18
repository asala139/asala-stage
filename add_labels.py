import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer


os.system('clear')
__DIR__ = os.path.dirname(os.path.abspath(__file__))


def load_complete_dataframe(df_input):
    df_global = []
    for f in os.listdir(df_input):
        year = int(f.split('.')[0]) # estrai anno

        df_years = pd.read_csv(df_input + f, index_col=False)

        df_years['years'] = year # aggiungi colonna con anno
    
        pd.set_option('future.no_silent_downcasting', True)

        df_years.replace("n.d.", np.nan, inplace=True)
        df_years = df_years.infer_objects()
    
        # df_years = df_years[df_years['Numero di anni disponibili'] == 10]
        if len(df_global) == 0:
            df_global = df_years
        else:
            df_global = pd.concat([df_global, df_years])

    # df_global.drop('Unnamed: 0', inplace=True, axis=1)
    df_global.reset_index(drop=True, inplace=True)
    
    return df_global

def add_labels_to_dataset(df_global, row_labels='Utile/perdita di esercizio [utile netto]', company_column='Ragione socialeCaratteri latini', year_column='years'):
    dfs = [] 
    for company in df_global[company_column].unique(): # per ogni azienda
        df = df_global[df_global[company_column] == company].copy()

        df = df.sort_values(by=year_column) # ordina per anno

        last_utile = None
        labels = []

        for _, row in df.iterrows():
            current_utile = row[row_labels]
            if last_utile is None:
                last_utile = current_utile
                labels.append(-1) # prima riga
                continue
            if current_utile >= last_utile:
                labels.append(1) # utile in crescita
            else:
                labels.append(0) # utile in calo
            last_utile = current_utile
        df['labels ' + row_labels] = labels
        dfs.append(df)
    
    df_global = pd.concat(dfs)

    last_year = df_global[year_column].max()
    df_global = df_global[df_global[year_column] != last_year]

    return df_global

def clean_columns(df, columns): # sostituisce valori nulli 
    zero_imputer = SimpleImputer(strategy='constant', fill_value=0)
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(["n.d.", "n.s.", "n.a."], np.nan).astype(float)
            df[[col]] = zero_imputer.fit_transform(df[[col]])
    return df

def process_dataset(df_input, output_file, columns_to_clean):
    df_global = load_complete_dataframe(df_input)

    df_global = clean_columns(df_global, columns_to_clean)

    for column in columns_to_clean:
        df_global = add_labels_to_dataset(df_global, row_labels=column)

    df_global.to_csv(output_file, index=False)
    print(f"Dataset elaborato e salvato in: {output_file}")

#####
columns_to_clean = ['Utile/perdita di esercizio [utile netto]',
                    'Rendimento del capitale investito (ROCE) - Netto',
                    'Redditività del capitale proprio (ROE) - Netto']

process_dataset(df_input=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/ITA/",
                output_file=__DIR__ + "/labels/ITA_labels.csv",
                columns_to_clean=columns_to_clean)

process_dataset(df_input=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/EU_27/by_years/",
                output_file=__DIR__ + "/labels/EU_labels.csv",
                columns_to_clean=columns_to_clean)

process_dataset(df_input=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/USA/by_years/",
                output_file=__DIR__ + "/labels/USA_labels.csv",
                columns_to_clean=columns_to_clean)