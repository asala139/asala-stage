"""
# SENZA IMPUTAZIONE
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("/Users/anna/python/with_venv/stage/Datasets/Orbis-Dataset/Numeric/ITA/2022.csv")

plt.hist(df["Risultato operativo [EBIT]"])

plt.show()
"""

#CON IMPUTAZIONE
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

#file path
df = pd.read_csv("/Users/anna/python/with_venv/stage/Datasets/Orbis-Dataset/Numeric/ITA/2022.csv")

#target
#utile netto
#col = "Utile/perdita di esercizio [utile netto]"
#EBITDA
col = "EBITDA"
#ROA
#col = "Redditività del totale Attivo (ROA) - Netto"

#sostituire n.d. con NaN
df[col] = df[col].replace("n.d.", np.nan).astype(float)

#3 metodi 
df_zero = df.copy()
df_mean = df.copy()
df_median = df.copy()

#imputazione con 0
zero_imputer = SimpleImputer(strategy='constant', fill_value=0)
df_zero[[col]] = zero_imputer.fit_transform(df_zero[[col]])

#imputazione con media
mean_imputer = SimpleImputer(strategy='mean')
df_mean[[col]] = mean_imputer.fit_transform(df_mean[[col]])

#imputazione con mediana
median_imputer = SimpleImputer(strategy='median')
df_median[[col]] = median_imputer.fit_transform(df_median[[col]])

#confronto grafici
plt.subplot(1, 3, 1)
plt.hist(df_zero[col], color='blue')
plt.title("Imputazione con 0")

plt.subplot(1, 3, 2)
plt.hist(df_mean[col], color='green')
plt.title("Imputazione con media")

plt.subplot(1, 3, 3)
plt.hist(df_median[col], color='red')
plt.title("Imputazione con mediana")

plt.show()
