import numpy as np

def get_top_n(arr,n):
    sorted_index_array = np.argsort(arr)
    sorted_array = arr[sorted_index_array]
    rslt = sorted_array[-n:]
    return rslt, sorted_index_array[-n:]

def get_last_n(arr,n):
    sorted_index_array = np.argsort(arr)
    sorted_array = arr[sorted_index_array]
    rslt = sorted_array[:n]
    return rslt, sorted_index_array[:n]
