import pandas as pd
import numpy as np
import os
__DIR__ = os.path.dirname(os.path.abspath(__file__))

def remove_non_numeric_features(df_global, column_names = []):
    if len(column_names) == 0:
        column_names = ["name", "fiscal_code", "city", "CCIAA_code", "Stato giuridico",
                                "Forma giuridica", "Data di costituzione", "Data di costituzione",
                                "Ultimo modello di contabilità - Bilancio", "Indicatore d'Indipendenza BvD",
                                "N° azionisti registrati", "N° partecipate registrate", "Principale Borsa",
                                "Descrizione attività italiano", "Ateco 2007", "SAE code",
                                "Descrizione attività inglese",
                                "RAE code", "Numero di anni disponibili", 'No of companies in corporate group',
                                '  Dipendenti']
    for c in column_names:
        try:
            df_global = df_global.drop([c], axis=1)
        except:
            #print(c + " not found in dataframe!")
            continue
    return df_global

def substitution_with_zero(df_global):
    df_global = df_global.replace(['n.d.', 'n.s.', 'n.a.'], np.nan)
    if "Periodo di competenza" in df_global.columns:
        df_global = df_global[df_global["Periodo di competenza"].notnull()]
    df_global = df_global.fillna(0)
    return df_global

def substitution_with_mean(df_global):
    df_global = df_global.replace(['n.d.', 'n.s.', 'n.a.'], np.nan)
    
    if "Periodo di competenza" in df_global.columns:
        df_global = df_global[df_global["Periodo di competenza"].notnull()]

    exclude_object_cols = [
        "Data chiusura", "Anno fiscale", "Trimestre", "Periodo di competenza",
        "Stato revisione/audit", "Status bilancio", "Principi contabili",
        "Fonte dati di bilancio", "Unità originale", "Valuta originale"
    ]

    for col in df_global.select_dtypes(include=['object']).columns:
        if col not in exclude_object_cols:
            try:
                df_global[col] = pd.to_numeric(df_global[col], errors='raise')
            except:
                pass

    for col in df_global.select_dtypes(include=[np.number]).columns:
        mean_value = df_global[col].mean(skipna=True)
        df_global[col] = df_global[col].fillna(mean_value)

    df_global = df_global.fillna(0)  # perchè alcune colonne sono tutte NaN, quindi la media risulta NaN

    return df_global

def substitution_with_median(df_global):
    df_global = df_global.replace(['n.d.', 'n.s.', 'n.a.'], np.nan)
    
    if "Periodo di competenza" in df_global.columns:
        df_global = df_global[df_global["Periodo di competenza"].notnull()]

    exclude_object_cols = [
        "Data chiusura", "Anno fiscale", "Trimestre", "Periodo di competenza",
        "Stato revisione/audit", "Status bilancio", "Principi contabili",
        "Fonte dati di bilancio", "Unità originale", "Valuta originale"
    ]

    for col in df_global.select_dtypes(include=['object']).columns:
        if col not in exclude_object_cols:
            try:
                df_global[col] = pd.to_numeric(df_global[col], errors='raise')
            except:
                pass

    for col in df_global.select_dtypes(include=[np.number]).columns:
        median_value = df_global[col].median(skipna=True)
        df_global[col] = df_global[col].fillna(median_value)

    df_global = df_global.fillna(0)  # perchè alcune colonne sono tutte NaN, quindi la media risulta NaN

    return df_global


def sustitute_inf(df_global):
    df_global.replace([np.inf, -np.inf], 0, inplace=True)
    return df_global

def change_feature_type(df_global):
    for c in df_global.columns:
        if c == 'labels' or c == 'years':
            df_global[c] = df_global[c].astype("int")
        else:
            try:
                df_global[c] = df_global[c].astype("float64")
            except Exception as exc:
                #print(c)
                continue
    return df_global

def group_dataset_three_years(df_global):
    df_2013 = df_global[df_global['years'] == 2013]
    df_2014 = df_global[df_global['years'] == 2014]
    df_2015 = df_global[df_global['years'] == 2015]
    df_one = df_2013.merge(df_2014, on='name', how='inner', suffixes=('_x', '_y'))
    df_one = df_one.merge(df_2015, on='name', how='inner', suffixes=('', '_z'))

    df_2014 = df_global[df_global['years'] == 2014]
    df_2015 = df_global[df_global['years'] == 2015]
    df_2016 = df_global[df_global['years'] == 2016]
    df_two = df_2014.merge(df_2015, how='inner', on='name', suffixes=('_x', '_y'))
    df_two = df_two.merge(df_2016, on='name', how='inner', suffixes=('', '_z'))

    df_2015 = df_global[df_global['years'] == 2015]
    df_2016 = df_global[df_global['years'] == 2016]
    df_2017 = df_global[df_global['years'] == 2017]
    df_three = df_2015.merge(df_2016, how='inner', on='name', suffixes=('_x', '_y'))
    df_three = df_three.merge(df_2017, on='name', how='inner', suffixes=('', '_z'))

    df_2016 = df_global[df_global['years'] == 2016]
    df_2017 = df_global[df_global['years'] == 2017]
    df_2018 = df_global[df_global['years'] == 2018]
    df_four = df_2016.merge(df_2017, how='inner', on='name', suffixes=('_x', '_y'))
    df_four = df_four.merge(df_2018, on='name', how='inner', suffixes=('', '_z'))

    df_train = pd.concat([df_one, df_two, df_three, df_four])

    df_train = df_train.drop(["labels_x", "labels_y"], axis=1)
    df_train = df_train.rename(columns={"labels_z": "labels"})

    df_2019 = df_global[df_global['years'] == 2019]
    df_2020 = df_global[df_global['years'] == 2020]
    df_2021 = df_global[df_global['years'] == 2021]

    df_test = df_2019.merge(df_2020, how='inner', on='name', suffixes=('_x', '_y'))
    df_test = df_test.merge(df_2021, on='name', how='inner', suffixes=('', '_z'))

    df_test = df_test.drop(["labels_x", "labels_y"], axis=1)
    df_test = df_test.rename(columns={"labels_z": "labels"})
    return df_train, df_test

def remove_non_numeric_feature_three_years(df_global, column_names_one = [], column_names_two = [], column_names_three = []):
    if len(column_names_one) == 0:
        column_names_one = ["name", "fiscal_code_x", "city_x", "CCIAA_code_x", "Stato giuridico_x",
                            "Forma giuridica_x", "Data di costituzione_x", "Data di costituzione_x",
                            "Ultimo modello di contabilità - Bilancio_x", "Indicatore d'Indipendenza BvD_x",
                            "N° azionisti registrati_x", "N° partecipate registrate_x", "Principale Borsa_x",
                            "Descrizione attività italiano_x", "Ateco 2007_x", "SAE code_x",
                            "Descrizione attività inglese_x",
                            "RAE code_x", "Numero di anni disponibili_x", 'No of companies in corporate group_x',
                            '  Dipendenti_x', 'years_x']
    df_global = remove_non_numeric_features(df_global, column_names_one)
    if len(column_names_two) == 0:
        column_names_two = ["fiscal_code_y", "city_y", "CCIAA_code_y", "Stato giuridico_y",
                            "Forma giuridica_y", "Data di costituzione_y", "Data di costituzione_y",
                            "Ultimo modello di contabilità - Bilancio_y", "Indicatore d'Indipendenza BvD_y",
                            "N° azionisti registrati_y", "N° partecipate registrate_y", "Principale Borsa_y",
                            "Descrizione attività italiano_y", "Ateco 2007_y", "SAE code_y",
                            "Descrizione attività inglese_y",
                            "RAE code_y", "Numero di anni disponibili_y", 'No of companies in corporate group_y',
                            '  Dipendenti_y', 'years_y']
    df_global = remove_non_numeric_features(df_global, column_names_two)
    if len(column_names_three) == 0:
        column_names_three = ["fiscal_code", "city", "CCIAA_code", "Stato giuridico",
                            "Forma giuridica", "Data di costituzione", "Data di costituzione",
                            "Ultimo modello di contabilità - Bilancio", "Indicatore d'Indipendenza BvD",
                            "N° azionisti registrati", "N° partecipate registrate", "Principale Borsa",
                            "Descrizione attività italiano", "Ateco 2007", "SAE code", "Descrizione attività inglese",
                            "RAE code", "Numero di anni disponibili", 'No of companies in corporate group',
                            '  Dipendenti', 'years']
    df_global = remove_non_numeric_features(df_global, column_names_three)
    return df_global

def group_dataset_two_years(df_global):
    df_2013 = df_global[df_global['years'] == 2013]
    df_2014 = df_global[df_global['years'] == 2014]
    df_one = df_2013.merge(df_2014, on='name', how='inner')

    df_2015 = df_global[df_global['years'] == 2015]
    df_two = df_2014.merge(df_2015, how='inner', on='name')

    df_2016 = df_global[df_global['years'] == 2016]
    df_three = df_2015.merge(df_2016, how='inner', on='name')

    df_2017 = df_global[df_global['years'] == 2017]
    df_four = df_2016.merge(df_2017, how='inner', on='name')

    df_2018 = df_global[df_global['years'] == 2018]
    df_five = df_2017.merge(df_2018, how='inner', on='name')

    df_2019 = df_global[df_global['years'] == 2019]
    df_six = df_2018.merge(df_2019, how='inner', on='name')

    df_train = pd.concat([df_one, df_two, df_three, df_four, df_five, df_six])

    try:
        df_train = df_train.drop(["labels_x"], axis=1)
    except KeyError as exc:
        print(exc)
    try:
        df_train = df_train.drop(["labels_y"], axis=1)
    except KeyError as exc:
        print(exc)
    #df_train = df_train.rename(columns={"labels_y": "labels"})

    df_2020 = df_global[df_global['years'] == 2020]
    df_2021 = df_global[df_global['years'] == 2021]

    df_test = df_2020.merge(df_2021, how='inner', on='name')

    try:
        df_test = df_test.drop(["labels_x"], axis=1)
    except KeyError as exc:
        print(exc)
    try:
        df_test = df_test.drop(["labels_y"], axis=1)
    except KeyError as exc:
        print(exc)
    #df_test = df_test.drop(["labels_x"], axis=1)
    #df_test = df_test.rename(columns={"labels_y": "labels"})
    return df_train, df_test

def remove_non_numeric_feature_two_years(df_global, column_names_one = [], column_names_two = []):
    if len(column_names_one) == 0:
        column_names_one = ["name", "fiscal_code_x", "city_x", "CCIAA_code_x", "Stato giuridico_x",
                            "Forma giuridica_x", "Data di costituzione_x", "Data di costituzione_x",
                            "Ultimo modello di contabilità - Bilancio_x", "Indicatore d'Indipendenza BvD_x",
                            "N° azionisti registrati_x", "N° partecipate registrate_x", "Principale Borsa_x",
                            "Descrizione attività italiano_x", "Ateco 2007_x", "SAE code_x",
                            "Descrizione attività inglese_x",
                            "RAE code_x", "Numero di anni disponibili_x", 'No of companies in corporate group_x',
                            '  Dipendenti_x']
    df_global = remove_non_numeric_features(df_global, column_names_one)
    if len(column_names_two) == 0:
        column_names_two = ["fiscal_code_y", "city_y", "CCIAA_code_y", "Stato giuridico_y",
                            "Forma giuridica_y", "Data di costituzione_y", "Data di costituzione_y",
                            "Ultimo modello di contabilità - Bilancio_y", "Indicatore d'Indipendenza BvD_y",
                            "N° azionisti registrati_y", "N° partecipate registrate_y", "Principale Borsa_y",
                            "Descrizione attività italiano_y", "Ateco 2007_y", "SAE code_y",
                            "Descrizione attività inglese_y",
                            "RAE code_y", "Numero di anni disponibili_y", 'No of companies in corporate group_y',
                            '  Dipendenti_y', 'years_y']
    df_global = remove_non_numeric_features(df_global, column_names_two)
    return df_global

def remove_absence_value(df, threshold):
    column_names = []
    length = len(df)
    for i in range(len(df.columns)):
        c_name = df.columns[i]
        v_c = df[c_name].value_counts()
        v_c.index = v_c.index.map(str)
        try:
            n_d = v_c['n.d.']
            if type(n_d) == pd.Series:
                n_d = v_c.iloc[0]
        except KeyError as exc:
            n_d = 0
        try:
            n_s = v_c['n.s.']
            if type(n_s) == pd.Series:
                n_s = v_c.iloc[0]
        except KeyError as exc:
            n_s = 0
        try:
            n_a = v_c['n.a.']
            if type(n_a) == pd.Series:
                n_a = v_c.iloc[0]
        except KeyError as exc:
            n_a = 0
        try:
            zero = v_c['0']
            if type(zero) == pd.Series:
                zero = v_c.iloc[0]
        except KeyError as exc:
            zero = 0
        percentage = ((n_d + n_s + zero) / length) * 100
        if percentage > threshold:
            column_names.append(c_name)
    df = df.drop(column_names, axis=1)
    return df

def create_numeric_and_non_numeric(df_global, non_numeric_to_extend=['  Utile Netto', 'years']):
    column_names = []
    non_numeric_columns = []
    for c in df_global.columns:
        if c == '  Utile Netto':
            continue
        try:
            df_global[c] = df_global[c].astype(float)
            column_names.append(c)
        except ValueError as err:
            non_numeric_columns.append(c)
            #print(c)
            continue
    non_numeric_columns.extend(non_numeric_to_extend)
    numeric_value = df_global[column_names]
    non_numeric = df_global[non_numeric_columns]
    return numeric_value, non_numeric

def create_dataset_difference_by_years(numeric_value, non_numeric, years):
    new_dfs = []
    for y in years:
        df_this_year = numeric_value[numeric_value['years'] == y]
        new_year = y + 1
        df_next_year = numeric_value[numeric_value['years'] == new_year]
        df_this_year = df_this_year.drop(["years"], axis=1)
        df_next_year = df_next_year.drop(["years"], axis=1)
        if len(df_next_year) == 0:
            break
        values_next_year = df_next_year.values
        values_this_year = df_this_year.values
        new_values = np.zeros_like(values_next_year)
        for i in range(values_next_year.shape[1]):
            new_values[:, i] = ((values_next_year[:, i] - values_this_year[:, i]) / values_next_year[:, i]) * 100
        new_df_next_year = pd.DataFrame(new_values, columns=df_next_year.columns)
        new_dfs.append(new_df_next_year)

    df_global_new = pd.concat(new_dfs).reset_index()
    df_global_new = pd.concat([non_numeric, df_global_new], axis=1)
    return df_global_new

def normalize_by_value(numeric_value, non_numeric, years, column_name = '  Posizione finanziaria netta'):
    new_dfs = []
    for y in years:
        df_this_year = numeric_value[numeric_value['years'] == y]
        df_this_year = df_this_year.drop(["years"], axis=1)
        totale_attivita = df_this_year[column_name].to_numpy()
        #df_this_year = df_this_year.drop([column_name], axis=1)
        if len(df_this_year) == 0:
            break
        values_this_year = df_this_year.values
        new_values = np.zeros_like(values_this_year)
        for i in range(values_this_year.shape[1]):
            new_values[:, i] = values_this_year[:, i] / totale_attivita
        new_df_next_year = pd.DataFrame(new_values, columns=df_this_year.columns)
        new_dfs.append(new_df_next_year)

    df_global = pd.concat(new_dfs).reset_index()
    df_global = pd.concat([non_numeric, df_global], axis=1)
    return df_global
