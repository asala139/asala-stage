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

import shap

os.system('clear')
__DIR__ = os.path.dirname(os.path.abspath(__file__))

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)


#escludo primi 3 anni
#ds = ld.load_data_by_years_orbis(1998, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/ITA/")
ds = ld.load_data_by_years_orbis(1998, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/EU_27/by_years/")
#ds = ld.load_data_by_years_orbis(1998, 2023, dir=__DIR__ + "/Datasets/Orbis-Dataset/Numeric/USA/by_years/")

#ds = ds[ds['Inactive']=='No']

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

#Redditività del capitale proprio (ROE) - Netto
shap_mean_tables = {}

print("Gradient Boosting Classifier")
model = ensemble.GradientBoostingClassifier()
model, X_Test, y_test = ml_ut.train_model_for_xai(model, ds, label_feature='Redditività del capitale proprio (ROE) - Netto', test_year=2022)
#model, X_Test, y_test = ml_ut.train_model_for_xai(model, ds, label_feature='Rendimento del capitale investito (ROCE) - Netto', test_year=2022)
#model, X_Test, y_test = ml_ut.train_model_for_xai(model, ds, label_feature='Utile/perdita di esercizio [utile netto]', test_year=2022)

#explainer = shap.KernelExplainer(model.predict_proba, X_Test)
#explainer = shap.PermutationExplainer(model.predict_proba, X_Test)
explainer = shap.TreeExplainer(model)

shap_values = explainer(X_Test)
shap_array = shap_values.values

#mean_abs_shap = np.abs(shap_array[:, :, 1]).mean(axis=0)
mean_abs_shap = np.abs(shap_array).mean(axis=0)
shap_mean_tables['GradientBoosting'] = mean_abs_shap

model = ensemble.RandomForestClassifier()
print("Random Forest Classifier")
model, X_Test, y_test = ml_ut.train_model_for_xai(model, ds, label_feature='Redditività del capitale proprio (ROE) - Netto', test_year=2022)
#model, X_Test, y_test = ml_ut.train_model_for_xai(model, ds, label_feature='Rendimento del capitale investito (ROCE) - Netto', test_year=2022)
#model, X_Test, y_test = ml_ut.train_model_for_xai(model, ds, label_feature='Utile/perdita di esercizio [utile netto]', test_year=2022)

#explainer = shap.KernelExplainer(model.predict_proba, X_Test)
#explainer = shap.PermutationExplainer(model.predict_proba, X_Test)
explainer = shap.TreeExplainer(model)

shap_values = explainer(X_Test)
shap_array = shap_values.values

mean_abs_shap = np.abs(shap_array[:, :, 1]).mean(axis=0)
shap_mean_tables['RandomForest'] = mean_abs_shap
"""
model = nn.MLPClassifier()
print("Multilayer Perceptron Classifier")
model, X_Test, y_test = ml_ut.train_model_for_xai(model, ds, label_feature='Redditività del capitale proprio (ROE) - Netto', test_year=2022)
#model, X_Test, y_test = ml_ut.train_model_for_xai(model, ds, label_feature='Rendimento del capitale investito (ROCE) - Netto', test_year=2022)
#model, X_Test, y_test = ml_ut.train_model_for_xai(model, ds, label_feature='Utile/perdita di esercizio [utile netto]', test_year=2022)

#explainer = shap.KernelExplainer(model.predict_proba, X_Test)
explainer = shap.PermutationExplainer(model.predict_proba, X_Test)

shap_values = explainer(X_Test)
shap_array = shap_values.values

mean_abs_shap = np.abs(shap_array[:, :, 1]).mean(axis=0)
shap_mean_tables['MLP'] = mean_abs_shap
"""
shap_mean_matrix = pd.DataFrame.from_dict(shap_mean_tables, orient='index', columns=X_Test.columns)
shap_mean_matrix.to_csv("shap_means.csv")

#shap.plots.waterfall serve per visualizzare i valori SHAP per una singola istanza
#shap.plots.force serve per visualizzare i valori SHAP per una singola istanza in modo interattivo
#shap.plots.scatter serve per visualizzare i valori SHAP per tutte le istanze
#shap.plots.beeswarm serve per visualizzare i valori SHAP per tutte le istanze in modo interattivo
#shap.plots.bar serve per visualizzare i valori SHAP per tutte le istanze in modo interattivo

#shap.plots.bar(shap_values)
#shap.plots.beeswarm(shap_values)
#shap.summary_plot(shap_values, X_Test, plot_type="bar")
#shap.summary_plot(shap_values, X_Test) 
#shap.plots.scatter(shap_values)
# shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_Test.iloc[0], feature_names=X_Test.columns))

#shap_df = pd.DataFrame(shap_array[:, :, 1], columns=X_Test.columns)
#shap_df.to_csv("shap_values.csv", index=False)