

import os
from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
import fire

INTERIM_PATH = 'data/interim/'
REFINED_PATH = 'data/refined/'

def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
    
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
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")

def compute_share(dt):
    shares = dt.groupby(['dept_id', 'store_id', 'date'])['sales']\
               .sum().reset_index().rename(columns={'sales': 'gp_sales'})
    dt = dt.merge(shares, how='left')
    dt['sales'] = dt['sales']/dt['gp_sales']
    dt.drop(['gp_sales'], axis=1, inplace=True)
    return dt

def data_prep(horizon="validation", task="volume"):
    dataframe = pd.read_parquet(os.path.join(INTERIM_PATH, '%s_raw.parquet' % horizon))
    if task=="volume":
        create_fea(dataframe)
    elif task=="share":
        dataframe = compute_share(dataframe)
        create_fea(dataframe)
    else:
        raise ValueError('Wrong value for task.')

    dataframe.to_parquet(os.path.join(REFINED_PATH, '%s_%s_fe.parquet' % (horizon, task)))

if __name__ == '__main__':
  fire.Fire(data_prep)