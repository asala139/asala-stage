import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer


os.system('clear')
__DIR__ = os.path.dirname(os.path.abspath(__file__))


def load_complete_dataframe(df_input):
    df_global = []
    for f in os.listdir(df_input):
        # Estrai l'anno dal nome del file
        year = int(f.split('.')[0])

        df_years = pd.read_csv(df_input + f, index_col=False)

        df_years['years'] = year
    
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
    dfs = []  # Lista per memorizzare i DataFrame con le etichette

    # Itera su ogni azienda unica
    for company in df_global[company_column].unique():
        df = df_global[df_global[company_column] == company].copy()

        # Ordina il DataFrame per anno
        df = df.sort_values(by=year_column)

        # Inizializza variabili
        last_utile = None
        labels = []

        # Itera sulle righe del DataFrame
        for _, row in df.iterrows():
            current_utile = row[row_labels]

            # Gestione del primo valore
            if last_utile is None:
                last_utile = current_utile
                labels.append(-1)  # Etichetta iniziale
                continue

            # Confronta i valori e assegna l'etichetta
            if current_utile >= last_utile:
                labels.append(1)
            else:
                labels.append(0)

            # Aggiorna il valore precedente
            last_utile = current_utile

        # Aggiungi le etichette al DataFrame
        df['labels ' + row_labels] = labels
        dfs.append(df)

    # Concatena tutti i DataFrame
    df_global = pd.concat(dfs)

    # Rimuovi l'ultimo anno, se necessario
    last_year = df_global[year_column].max()
    df_global = df_global[df_global[year_column] != last_year]

    return df_global

def clean_columns(df, columns):
    # Pulisce le colonne specificate sostituendo valori non numerici con NaN e riempiendo i NaN con 0
    zero_imputer = SimpleImputer(strategy='constant', fill_value=0)
    for col in columns:
        if col in df.columns:
            # Sostituisci valori non numerici con NaN
            df[col] = df[col].replace(["n.d.", "n.s.", "n.a."], np.nan).astype(float)
            # Riempi i valori mancanti con 0
            df[[col]] = zero_imputer.fit_transform(df[[col]])
    return df

def process_dataset(df_input, output_file, columns_to_clean):
    # Carica il dataset
    df_global = load_complete_dataframe(df_input)

    # Pulisci le colonne specificate
    df_global = clean_columns(df_global, columns_to_clean)

    # Aggiungi le etichette
    for column in columns_to_clean:
        df_global = add_labels_to_dataset(df_global, row_labels=column)

    # Salva il risultato in un file CSV
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