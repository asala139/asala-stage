import pickle

import pandas as pd
import numpy as np
from balance_sheet_library.preprocessing import remove_non_numeric_features, remove_absence_value, substitution_with_zero, substitution_with_mean, substitution_with_median, change_feature_type
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def add_labels_to_dataset(df_global, row_labels='  Utile Netto'):
    df_global.sort_values(by='years', inplace=True)
    dfs = []
    for c in df_global['name'].unique():
        last_utile = -1
        df = df_global[df_global['name'] == c]
        df = df.sort_values('years')
        labels = []
        for i, row in df.iterrows():
            if last_utile == -1:
                if row[row_labels] == ["n.d.", "n.a.", "n.s."]:
                    last_utile = 0
                else:
                    last_utile = int(float(row[row_labels]))
                continue
            else:
                if row[row_labels] == ["n.d.", "n.a.", "n.s."]:
                    current_utile = 0
                else:
                    current_utile = int(float(row[row_labels]))
                if current_utile >= last_utile:
                    l = 1
                else:
                    l = 0
            labels.append(l)
            last_utile = current_utile
        labels.append(-1)
        df['labels'] = labels
        dfs.append(df)
    last_year = np.sort(df_global['years'].unique())[-1]
    df_global = pd.concat(dfs)
    df_global = df_global.drop(df_global[df_global['years'] == last_year].index)
    return df_global


def add_labels_without_int_to_dataset(df_global, row_labels='  Utile Netto'):
    df_global.sort_values(by='years', inplace=True)
    dfs = []
    for c in df_global['name'].unique():
        last_utile = ''
        df = df_global[df_global['name'] == c]
        df = df.sort_values('years')
        labels = []
        for i, row in df.iterrows():
            if last_utile == '':
                if row[row_labels] == ["n.d.", "n.a.", "n.s."]:
                    last_utile = 0
                else:
                    last_utile = float(row[row_labels])
                continue
            else:
                if row[row_labels] == ["n.d.", "n.a.", "n.s."]:
                    current_utile = 0
                else:
                    current_utile = float(row[row_labels])
                if current_utile >= last_utile:
                    l = 1
                else:
                    l = 0
            labels.append(l)
            last_utile = current_utile
        labels.append(-1)
        df['labels'] = labels
        dfs.append(df)
    last_year = np.sort(df_global['years'].unique())[-1]
    df_global = pd.concat(dfs)
    df_global = df_global.drop(df_global[df_global['years'] == last_year].index)
    return df_global

def base_preprocessing_for_classification(df_global, insertion=0):
    df_global = df_global.loc[:, ~df_global.columns.duplicated()] #eliminazione dei duplicati
    if insertion == 0:
        df_global = substitution_with_zero(df_global) #da cambiare
    elif insertion == 1:
        df_global = substitution_with_mean(df_global)
    elif insertion == 2:
        df_global = substitution_with_median(df_global)
    df_global = change_feature_type(df_global)
    return df_global

def create_test_set(ds, test_year):
    X_Test = ds[ds['years'] == test_year]
    y_test = X_Test['labels']
    X_Test = X_Test.drop(["years", "labels"], axis=1)
    X_Test = remove_non_numeric_features(X_Test)
    index_years = ds[ds['years'] == test_year].index
    ds.drop(index_years, inplace=True)
    X_Train = ds
    y_train = X_Train["labels"].to_numpy()
    X_Train = X_Train.drop(["years", "labels"], axis=1)
    X_Train = remove_non_numeric_features(X_Train)
    return X_Train, y_train, X_Test, y_test


def base_experiments(model, ds, remove_zero_with_threshold=False, threshold=10, test_year=2021, save_probability=True,
                     probability_path_file="prova/prova.csv", reduced_set_of_features = [], use_less_features =False,
                     label_feature='  Utile Netto', generate_labels=True, cast_int=False):
    if use_less_features:
        ds = ds[reduced_set_of_features]
    if generate_labels and cast_int:
        ds = add_labels_to_dataset(ds, label_feature)
    elif generate_labels:
        ds = add_labels_without_int_to_dataset(ds, label_feature)
    X_Test = ds[ds['years'] == test_year]
    y_test = X_Test['labels']
    X_Test = X_Test.drop(["years", "labels"], axis=1)
    X_Test = remove_non_numeric_features(X_Test)
    index_years = ds[ds['years'] == test_year].index
    ds.drop(index_years, inplace=True)
    X_Train = ds
    y_train = X_Train["labels"].to_numpy()
    X_Train = X_Train.drop(["years", "labels"], axis=1)
    X_Train = remove_non_numeric_features(X_Train)

    #filtro righe con label -1
    mask_train = y_train != -1
    X_Train = X_Train[mask_train]
    y_train = y_train[mask_train]

    mask_test = y_test != -1
    X_Test = X_Test[mask_test]
    y_test = y_test[mask_test]
    #fine filtro

    if remove_zero_with_threshold:
        X_Train = remove_absence_value(X_Train, threshold)
        X_Test = remove_absence_value(X_Test, threshold)
    model.fit(X_Train, y_train)

    y_pred = model.predict(X_Test)

    print(model)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("roc-auc score:")
    print(roc_auc_score(y_test, y_pred))

    if save_probability:
        labels = np.sort(np.unique(y_test))
        prob = model.predict_proba(X_Test)
        df_prob = pd.DataFrame(prob, columns=labels)
        df_prob["real labels"] = y_test
        df_prob["predicted"] = y_pred
        df_prob.to_csv(probability_path_file)

def train_model_for_xai(model, ds, remove_zero_with_threshold=False, threshold=10, test_year=2021,
                        reduced_set_of_features = [], use_less_features =False, label_feature='  Utile Netto',
                        generate_labels=True, cast_int=False, save_model_pickle=False, model_path="prova/prova.pkl",
                        save_test_set=False, test_set_path="prova/testset.pkl", save_label_test=False,
                        labels_path='prova/labels.pkl'):
    if use_less_features:
        ds = ds[reduced_set_of_features]
    if generate_labels and cast_int:
        ds = add_labels_to_dataset(ds, label_feature)
    elif generate_labels:
        ds = add_labels_without_int_to_dataset(ds, label_feature)
    X_Test = ds[ds['years'] == test_year]
    y_test = X_Test['labels']
    X_Test = X_Test.drop(["years", "labels"], axis=1)
    X_Test = remove_non_numeric_features(X_Test)
    index_years = ds[ds['years'] == test_year].index
    ds.drop(index_years, inplace=True)
    X_Train = ds
    y_train = X_Train["labels"].to_numpy()
    X_Train = X_Train.drop(["years", "labels"], axis=1)
    X_Train = remove_non_numeric_features(X_Train)

    if remove_zero_with_threshold:
        X_Train = remove_absence_value(X_Train, threshold)
        X_Test = remove_absence_value(X_Test, threshold)
    model.fit(X_Train, y_train)

    if save_model_pickle:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    if save_test_set:
        with open(test_set_path, 'wb') as f:
            pickle.dump(X_Test, f)
    if save_label_test:
        with open(labels_path, 'wb') as f:
            pickle.dump(y_test, f)

    return model, X_Test, y_test
