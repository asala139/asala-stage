import numpy as np
#from utility import get_top_n, get_last_n
import balance_sheet_library.utility as utility

def generate_heatmaps(shap_values):
    hm = []
    for s in shap_values:
        class_0 = []
        class_1 = []
        for f in s:
            class_0.append(f[0])
            class_1.append(f[1])
        hm_int = np.array([np.array(class_0), np.array(class_1)])
        hm.append(hm_int)
    return hm

def count_for_financial_part(hm,n_top=10):
    count_profilo_finanziario_0 = 0
    count_stato_patrimoniale_0 = 0
    count_conto_economico_0 = 0
    count_indici_0 = 0

    count_profilo_finanziario_1 = 0
    count_stato_patrimoniale_1 = 0
    count_conto_economico_1 = 0
    count_indici_1 = 0

    for h in hm:
        top_ten, top_ten_index = utility.get_top_n(h[0], n_top)
        for i in top_ten_index:
            if i < 14:
                count_profilo_finanziario_0 += 1
            elif i < 177 and i > 13:
                count_stato_patrimoniale_0 += 1
            elif i < 256 and i > 176:
                count_conto_economico_0 += 1
            else:
                count_indici_0 += 1
        top_ten, top_ten_index = utility.get_top_n(h[1], n_top)
        for i in top_ten_index:
            if i < 14:
                count_profilo_finanziario_1 += 1
            elif i < 177 and i > 13:
                count_stato_patrimoniale_1 += 1
            elif i < 256 and i > 176:
                count_conto_economico_1 += 1
            else:
                count_indici_1 += 1
    array_plot_0 = [count_profilo_finanziario_0, count_stato_patrimoniale_0, count_conto_economico_0, count_indici_0]
    array_plot_1 = [count_profilo_finanziario_1, count_stato_patrimoniale_1, count_conto_economico_1, count_indici_1]
    return array_plot_0, array_plot_1

def create_top_n(hm,n_top=10):
    top_ten_class0 = []
    top_ten_class1 = []

    top_ten_index_class0 = []
    top_ten_index_class1 = []

    for h in hm:
        top_ten, top_ten_index = utility.get_top_n(h[0], n_top)
        top_ten_class0.extend(top_ten)
        top_ten_index_class0.extend(top_ten_index)
        top_ten, top_ten_index = utility.get_top_n(h[1], n_top)
        top_ten_class1.extend(top_ten)
        top_ten_index_class1.extend(top_ten_index)
    top_ten_class0 = np.array(top_ten_class0)
    top_ten_class1 = np.array(top_ten_class1)
    unique_class_0, counts_class_0 = np.unique(top_ten_index_class0, return_counts=True)
    unique_class_1, counts_class_1 = np.unique(top_ten_index_class1, return_counts=True)

    return unique_class_0, counts_class_0, unique_class_1, counts_class_1


def create_last_n(hm, n_top=10):
    top_ten_class0 = []
    top_ten_class1 = []

    top_ten_index_class0 = []
    top_ten_index_class1 = []

    for h in hm:
        top_ten, top_ten_index = utility.get_last_n(h[0], n_top)
        top_ten_class0.extend(top_ten)
        top_ten_index_class0.extend(top_ten_index)
        top_ten, top_ten_index = utility.get_last_n(h[1], n_top)
        top_ten_class1.extend(top_ten)
        top_ten_index_class1.extend(top_ten_index)
    top_ten_class0 = np.array(top_ten_class0)
    top_ten_class1 = np.array(top_ten_class1)
    unique_class_0, counts_class_0 = np.unique(top_ten_index_class0, return_counts=True)
    unique_class_1, counts_class_1 = np.unique(top_ten_index_class1, return_counts=True)

    return unique_class_0, counts_class_0, unique_class_1, counts_class_1