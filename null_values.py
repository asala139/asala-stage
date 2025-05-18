import numpy as np
import pandas as pd
import os

os.system('clear')
__DIR__ = os.path.dirname(os.path.abspath(__file__))

# funzione per caricare il dataframe
def load_complete_dataframe(df_input):
    df_global = []
    for f in os.listdir(df_input):
        df_years = pd.read_csv(df_input + f, index_col=False)
    
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

# funzione per calcolare i valori nulli
def calculate_null_values(df_global, output_file, num_companies, num_years, start_year):
    # dizionario per risultati
    results = {"Nome Colonna": []}

    # definzione colonne
    for i in range(num_years):
        results[f"# Nulli {i + start_year}"] = []
        results[f"% Nulli {i + start_year}"] = []

    for column in df_global.columns:
        results["Nome Colonna"].append(column)

        # calcolo valori nulli per ogni anno
        for i in range(num_years):
            start_range = i * num_companies
            end_range = start_range + num_companies - 1

            # prende un anno alla volta selezionando le righe con iloc
            subset_df = df_global.iloc[start_range:end_range + 1]

            # calcolo valori nulli
            missing_values = subset_df[column].isna().sum()
            # calcolo percentuale valori nulli
            missing_percentage = round((missing_values / num_companies) * 100, 2)

            # aggiornamento dizionario
            results[f"# Nulli {i + start_year}"].append(missing_values)
            results[f"% Nulli {i + start_year}"].append(missing_percentage)

    # creazione dataframe e salvataggio su file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file)

################################################################

# ITA
df_input_ita = __DIR__ + "/Datasets/Orbis-Dataset/Numeric/ITA/"
df_global_ita = load_complete_dataframe(df_input_ita)
output_file_ita = __DIR__ + "/null_values/null_values_ita.csv"
calculate_null_values(df_global_ita, output_file_ita, num_companies=362, num_years=30, start_year=1995)

# EU
df_input_eu = __DIR__ + "/Datasets/Orbis-Dataset/Numeric/EU_27/by_years/"
df_global_eu = load_complete_dataframe(df_input_eu)
output_file_eu = __DIR__ + "/null_values/null_values_eu.csv"
calculate_null_values(df_global_eu, output_file_eu, num_companies=2000, num_years=30, start_year=1995)

# USA
df_input_usa = __DIR__ + "/Datasets/Orbis-Dataset/Numeric/USA/by_years/"
df_global_usa = load_complete_dataframe(df_input_usa)
output_file_usa = __DIR__ + "/null_values/null_values_usa.csv"
calculate_null_values(df_global_usa, output_file_usa, num_companies=2000, num_years=30, start_year=1995)
