from multiprocessing import Pool  
import numpy as np
import pandas as pd
import os
import sys
import gc
import time
import warnings
import pickle
import random
import shutil
from datetime import datetime, timedelta


CAL_DTYPES = {"event_name_1": "category",
              "event_name_2": "category",
              "event_type_1": "category",
              "event_type_2": "category",
              "weekday": "category",
              'wm_yr_wk': 'int16',
              "wday": "int16",
              "month": "int16",
              "year": "int16",
              "snap_CA": "float32",
              'snap_TX': 'float32',
              'snap_WI': 'float32'}

PRICE_DTYPES = {"store_id": "category",
                "item_id": "category",
                "wm_yr_wk": "int16",
                "sell_price": "float32"}


# training constants 
use_sampled_ds = False # does not work yet  
h = 28 
max_lags = 366
tr_last = 1913#1941 for the new cutoff
fday = datetime(2016,4, 25) 
fday

def create_dt(is_train=True,
              nrows=None,
              first_day=1200,
              is_eval=False):
    prices = pd.read_csv("./data/raw/sell_prices.csv", dtype=PRICE_DTYPES)
    if is_eval:
        tr_last=1941
    else:
        tr_last=1913
        
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()

    cal = pd.read_csv("./data/raw/calendar.csv", dtype=CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()

    start_day = max(1 if is_train else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day, tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
    dtype = {numcol: "float32" for numcol in numcols}
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("./data/raw/sales_train_validation.csv",
                     nrows=nrows, usecols=catcols + numcols, dtype=dtype)

    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()

    increasing_term = dt.groupby(['dept_id', 'store_id'])[numcols].sum()
    increasing_term = (increasing_term.T -
                       increasing_term.T.shift(28))/increasing_term.T.shift(28)
    increasing_term = increasing_term.reset_index(drop=True).iloc[-365:, :]
    rates = increasing_term[increasing_term.abs() < 1].mean()+1
    rates = rates.reset_index().rename(columns={0: 'rate'})

    if not is_train:
        for day in range(tr_last+1, tr_last + 2*h + 1):
            dt[f"d_{day}"] = np.nan

    dt = pd.melt(dt,
                 id_vars=catcols,
                 value_vars=[col for col in dt.columns if col.startswith("d_")],
                 var_name="d",
                 value_name="sales")

    dt = dt.merge(cal, on="d", copy=False)
    dt = dt.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
    dt = dt.merge(rates, how='left')

    return dt

def create_features(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id", "sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby(
                "id")[lag_col].transform(lambda x: x.rolling(win).mean())

    date_features = {

        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(
                dt["date"].dt, date_feat_func).astype("int16")
           
           
def preprocess_interim(path_raw='./data/raw/',
                       path_interim='./data/interim/',
                       FIRST_DAY=1,
                      is_eval=False):
    sales_train_validation = pd.read_csv(path_raw +'sales_train_validation.csv')
    increasing_term = sales_train_validation.groupby(['dept_id', 'store_id'])\
                [['d_%s' % c for c in range(1,1914)]].sum()

    increasing_term = (increasing_term.T - increasing_term.T.shift(28))/increasing_term.T.shift(28)
    increasing_term = increasing_term.reset_index(drop=True).iloc[-365:,:]

    rates = increasing_term[increasing_term.abs()<1].mean()+1
    rates = rates.reset_index().rename(columns={0:'rate'})
    
    df = create_dt(is_train=True, first_day= FIRST_DAY,is_eval=is_eval)
    create_features(df)
    df.dropna(inplace = True)
    gc.collect()
    
    print('----> save the dataframe')
    if is_eval:
        df.to_csv(path_interim+'df_tf_'+'eval'+'_.gz', chunksize=100000
         , compression='gzip'
         , encoding='utf-8')
    else:
        df.to_csv(path_interim+'df_tf_'+'valid'+'_.gz', chunksize=100000
         , compression='gzip'
         , encoding='utf-8')
    

    