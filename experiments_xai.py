import sys
sys.path.insert(1, '../')

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
import balance_sheet_library.load_data as ld
import balance_sheet_library.ml_utility as ml_ut
import balance_sheet_library.preprocessing as prec
import numpy as np
import random
import os

os.system('clear')
__DIR__ = os.path.dirname(os.path.abspath(__file__))

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

#escludo primi 3 anni
#ds = ld.load_data_by_years_orbis(1998, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/ITA/")
#ds = ld.load_data_by_years_orbis(1998, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/EU_27/by_years/")
ds = ld.load_data_by_years_orbis(1998, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/USA/by_years/")

ds = ml_ut.base_preprocessing_for_classification(ds, insertion=1) #mean

import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import sklearn.svm as svm
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import matplotlib.pyplot as plt


ds.rename({'Ragione socialeCaratteri latini':'name'}, inplace=True, axis=1)

columns_remove = []
for c in ds.columns:
    if type(ds[c]) == pd.DataFrame:
        print(c)
        columns_remove.append(c)
        continue
    if ds[c].dtype == object:
        if c == 'name':
            continue
        columns_remove.append(c)

columns_remove.append('Anno fiscale')
columns_remove.append('Trimestre')
columns_remove.append('Periodo di competenza')
ds = prec.remove_non_numeric_features(ds, columns_remove)

# caratteristiche più importanti #
#df_importance = pd.read_csv(__DIR__ + "/xai_results/ITA-shap_means.csv", sep=';')
#df_importance = pd.read_csv(__DIR__ + "/xai_results/EU-shap_means.csv", sep=';')
df_importance = pd.read_csv(__DIR__ + "/xai_results/USA-shap_means.csv", sep=';')

df_importance.set_index(['label', 'explainer', 'model'], inplace=True)
#calcolo delle colonne da tenere in considerazione

print("Labels disponibili:", df_importance.index.get_level_values('label').unique().tolist())
print("Explainers disponibili:", df_importance.index.get_level_values('explainer').unique().tolist())
print("Models disponibili:", df_importance.index.get_level_values('model').unique().tolist())

def get_top_features(df_importance, label, explainer, model_name, top_n=10):
    try:
        row = df_importance.loc[(label, explainer, model_name)]
        if row.isnull().all():
            print(f"Riga trovata ma tutti valori NaN: ({label}, {explainer}, {model_name})")
            return []
        top_features = row.dropna().nlargest(top_n).index.tolist()
        #print(f"Top {top_n} features per ({label}, {explainer}, {model_name}): {top_features}")
        return top_features
    except KeyError:
        print(f"Riga non trovata: ({label}, {explainer}, {model_name})")
        return []

###

print(ds.shape)

class_map = {
    'GradientBoostingClassifier': ensemble.GradientBoostingClassifier,
    'RandomForestClassifier': ensemble.RandomForestClassifier,
    'MLPClassifier': nn.MLPClassifier
}

tasks = [
    'ROE',
    'ROCE',
    'Utile Netto'
]

label_column_map = {
    'ROE': 'Redditività del capitale proprio (ROE) - Netto',
    'ROCE': 'Rendimento del capitale investito (ROCE) - Netto',
    'Utile Netto': 'Utile/perdita di esercizio [utile netto]'
}

explainer_model_pairs = [
    ('KernelExplainer', 'GradientBoostingClassifier'),
    ('PermutationExplainer', 'GradientBoostingClassifier'),
    ('TreeExplainer', 'GradientBoostingClassifier'),
    ('KernelExplainer', 'RandomForestClassifier'),
    ('PermutationExplainer', 'RandomForestClassifier'),
    ('TreeExplainer', 'RandomForestClassifier'),
    ('KernelExplainer', 'MLPClassifier'),
    ('PermutationExplainer', 'MLPClassifier'),
]

for label in tasks:
    print(f"\n\033[1m{label}\033[0m")
    for explainer, model_name in explainer_model_pairs:
        print(f"--- Eseguo: Label={label}, Explainer={explainer}, Model={model_name} ---")
        model_class = class_map[model_name]
        #top_features = get_top_features(df_importance, label, explainer, model_name, top_n=10)
        top_features = get_top_features(df_importance, label, explainer, model_name, top_n=20)
        #top_features = get_top_features(df_importance, label, explainer, model_name, top_n=50)
        if not top_features:
            print(f"Skip missing features: ({label}, {explainer}, {model_name})")
            continue
        target_column = label_column_map[label]

        columns_to_keep = set(top_features + [target_column, 'name', 'years'])
        columns_to_keep = [col for col in ds.columns if col in columns_to_keep]

        ds_subset = ds[columns_to_keep]
        model = model_class()
        ml_ut.base_experiments(model, ds_subset, label_feature=target_column, save_probability=False, test_year=2022)