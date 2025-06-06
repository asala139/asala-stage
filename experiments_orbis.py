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

#tutti gli anni
#ds = ld.load_data_by_years_orbis(1995, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/ITA/")
#ds = ld.load_data_by_years_orbis(1995, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/EU_27/by_years/")
#ds = ld.load_data_by_years_orbis(1995, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/USA/by_years/")

#escludo primi 3 anni
#ds = ld.load_data_by_years_orbis(1998, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/ITA/")
#ds = ld.load_data_by_years_orbis(1998, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/EU_27/by_years/")
ds = ld.load_data_by_years_orbis(1998, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/USA/by_years/")

#ds = ds[ds['Inactive']=='No']

#ds = ml_ut.base_preprocessing_for_classification(ds) #zero
ds = ml_ut.base_preprocessing_for_classification(ds, insertion=1) #mean
#ds = ml_ut.base_preprocessing_for_classification(ds, insertion=2) #median

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

### null values ###
#nulls_df = pd.read_csv(__DIR__ + "/null_values/null_values_ita.csv") #ita
#nulls_df = pd.read_csv(__DIR__ + "/null_values/null_values_eu.csv") #eu
nulls_df = pd.read_csv(__DIR__ + "/null_values/null_values_usa.csv") #usa

#calcolo delle colonne da eliminare
percent_cols = [col for col in nulls_df.columns if col.startswith('% Nulli') and int(col.split()[-1]) >= 1998]
columns_to_drop = nulls_df[nulls_df[percent_cols].gt(90).any(axis=1)]['Nome Colonna'].tolist()
#columns_to_drop = nulls_df[nulls_df[percent_cols].gt(75).any(axis=1)]['Nome Colonna'].tolist()
#columns_to_drop = nulls_df[nulls_df[percent_cols].gt(50).any(axis=1)]['Nome Colonna'].tolist()
#columns_to_drop = nulls_df[nulls_df[percent_cols].gt(25).any(axis=1)]['Nome Colonna'].tolist()

#elimino le colonne
ds = ds.drop(columns=[col for col in columns_to_drop if col in ds.columns])
######

print(ds.shape)

#Redditività del capitale proprio (ROE) - Netto
print('\033[1m' + "\nRedditività del capitale proprio (ROE) - Netto" + '\033[0m')
model = ensemble.GradientBoostingClassifier()
ml_ut.base_experiments(model, ds, label_feature='Redditività del capitale proprio (ROE) - Netto', save_probability=False, test_year=2022)

model = ensemble.RandomForestClassifier()
ml_ut.base_experiments(model, ds, label_feature='Redditività del capitale proprio (ROE) - Netto', save_probability=False, test_year=2022)

model = nn.MLPClassifier()
ml_ut.base_experiments(model, ds, label_feature='Redditività del capitale proprio (ROE) - Netto', save_probability=False, test_year=2022)

#Rendimento del capitale investito (ROCE) - Netto
print('\033[1m' + "\nRendimento del capitale investito (ROCE) - Netto" + '\033[0m')
model = ensemble.GradientBoostingClassifier()
ml_ut.base_experiments(model, ds, label_feature='Rendimento del capitale investito (ROCE) - Netto', save_probability=False, test_year=2022)

model = ensemble.RandomForestClassifier()   
ml_ut.base_experiments(model, ds, label_feature='Rendimento del capitale investito (ROCE) - Netto', save_probability=False, test_year=2022)

model = nn.MLPClassifier()
ml_ut.base_experiments(model, ds, label_feature='Rendimento del capitale investito (ROCE) - Netto', save_probability=False, test_year=2022)

#Utile/perdita di esercizio [utile netto]
print('\033[1m' + "\nUtile/perdita di esercizio [utile netto]" + '\033[0m')
model = ensemble.GradientBoostingClassifier()
ml_ut.base_experiments(model, ds, label_feature='Utile/perdita di esercizio [utile netto]', save_probability=False, test_year=2022)

model = ensemble.RandomForestClassifier()
ml_ut.base_experiments(model, ds, label_feature='Utile/perdita di esercizio [utile netto]', save_probability=False, test_year=2022)

model = nn.MLPClassifier()
ml_ut.base_experiments(model, ds, label_feature='Utile/perdita di esercizio [utile netto]', save_probability=False, test_year=2022)
