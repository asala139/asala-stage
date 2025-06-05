import pandas as pd

import numpy as np
import math

def behind_to_front(ds):
    ds_list = []
    for company in ds['name'].unique():
        ds_c = ds[ds['name'] == company]
        ds_c.sort_values('years', inplace=True, ascending=False)
        new_esg = []
        prec_v = -1
        for e in ds_c['esg']:
            if prec_v == -1:
                if e == np.nan:
                    new_esg.append(e)
                    continue
                else:
                    prec_v = e
                    new_esg.append(e)
            else:
                if math.isnan(e):
                    new_esg.append(prec_v)
                else:
                    new_esg.append(e)
                    prec_v = e
        ds_c['esg'] = new_esg
        ds_list.append(ds_c)

    ds = pd.concat(ds_list)
    return ds

def front_to_behind(ds):
    ds_list = []
    for company in ds['name'].unique():
        ds_c = ds[ds['name'] == company]
        ds_c.sort_values('years', inplace=True, ascending=True)
        new_esg = []
        prec_v = -1
        for e in ds_c['esg']:
            if prec_v == -1:
                if e == np.nan:
                    new_esg.append(e)
                    continue
                else:
                    prec_v = e
                    new_esg.append(e)
            else:
                if math.isnan(e):
                    new_esg.append(prec_v)
                else:
                    new_esg.append(e)
                    prec_v = e
        ds_c['esg'] = new_esg
        ds_list.append(ds_c)

    ds = pd.concat(ds_list)
    return ds