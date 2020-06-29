
import sys
from tf_pipeline.tf_utils import create_mlp_softmax

import fire
from tf_pipeline.conf import *
import pandas as pd
import os 
import numpy as np 
from datetime import datetime 


def create_dt(horizon="validation", tr_last=1913):
    prices = pd.read_csv(os.path.join(RAW_PATH, "sell_prices.csv"), dtype=PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()

    cal = pd.read_csv(os.path.join(RAW_PATH, "calendar.csv"), dtype=CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()

    numcols = [f"d_{day}" for day in range(1, tr_last + 1)]
    catcols = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]
    dtype = {numcol: "float32" for numcol in numcols}
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv(
        os.path.join(RAW_PATH, "sales_train_%s.csv" % horizon),
        usecols=catcols + numcols,
        dtype=dtype,
    )

    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()

    increasing_term = dt.groupby(["dept_id", "store_id"])[numcols].sum()
    increasing_term = (
                              increasing_term.T - increasing_term.T.shift(28)
                      ) / increasing_term.T.shift(28)
    increasing_term = increasing_term.reset_index(drop=True).iloc[-365:, :]
    rates = increasing_term[increasing_term.abs() < 1].mean() + 1
    rates = rates.reset_index().rename(columns={0: "rate"})

    for day in range(tr_last + 1, tr_last + 2 * 28 + 1):
        dt[f"d_{day}"] = np.nan

    dt = pd.melt(
        dt,
        id_vars=catcols,
        value_vars=[col for col in dt.columns if col.startswith("d_")],
        var_name="d",
        value_name="sales",
    )

    dt = dt.merge(cal, on="d", copy=False)
    dt = dt.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
    dt = dt.merge(rates, how="left")

    return dt

def compute_share(dt):
    shares = (
        dt.groupby(["dept_id", "store_id", "date"])["sales"]
        .sum()
        .reset_index()
        .rename(columns={"sales": "gp_sales"})
    )
    dt = dt.merge(shares, how="left")
    dt["sales"] = dt["sales"] / dt["gp_sales"]
    dt.drop(["gp_sales"], axis=1, inplace=True)
    return dt

def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id", "sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = (
                dt[["id", lag_col]]
                    .groupby("id")[lag_col]
                    .transform(lambda x: x.rolling(win).mean())
            )

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
    return dt

def train_and_pred(horizon="validation"):

    if horizon=="validation":
        tr_last = 1913
        fday = datetime(2016, 4, 25)
    elif horizon=="evaluation":
        tr_last = 1941
        fday = datetime(2016, 4, 25) + timedelta(days=28)
    else:
        raise ValueError('Wrong horizon arg.')

    dataframe = create_dt(horizon, tr_last)
    dataframe = compute_share(dataframe)
    dataframe = create_fea(dataframe)
    dataframe.dropna(inplace=True)
    

    list_preds = list()

    for _, df_gp in dataframe.groupby(['store_id', 'dept_id']):

        cat_feats = ['wday', 'quarter']

        n_items = len(df_gp['item_id'].drop_duplicates())    

        ids = df_gp[['id', 'item_id']].drop_duplicates()\
                                      .sort_values('item_id')['id']\
                                      .tolist()
        X = df_gp[
            ['d', 'item_id', 'wday', 'quarter', 'date', 'rmean_28_28', 'sales']
        ].pivot_table(index=['d', 'date', 'wday', 'quarter'],
                      columns=['item_id'],
                      values=['rmean_28_28', 'sales']).fillna(0)

        num_feats = ['_'.join(list(map(str, c))) for c in X.columns.tolist()]
        X.columns = num_feats

        target_feats = num_feats[n_items:]
        num_feats = num_feats[:n_items]
        X = X.reset_index()

        X_train = X[X['date']<fday][num_feats+cat_feats]
        X_test = X[X['date']>=fday][num_feats+cat_feats]

        input_dict_train = {'input_%s' % c: X_train[c] for c in num_feats+cat_feats}
        input_dict_test = {'input_%s' % c: X_test[c] for c in num_feats+cat_feats}

        cardinality = X[cat_feats].nunique()

        y_train = X[X['date']<fday][target_feats].values

        mlp = create_mlp_softmax(layers_list=[2048, 2048],
                                 output_count=n_items,
                                 cat_feats=cat_feats,
                                 cardinality=cardinality,
                                 num_feats=num_feats)



        training_params = {
                    'x': input_dict_train,
                    'y': y_train,
                    'batch_size': 128,
                    'epochs': 20,
                    'shuffle': True,
                }

        mlp.fit(**training_params)
        preds = mlp.predict(input_dict_test)
        preds = pd.DataFrame(preds,
                             index=['F%s' % c for c in range(1,29)],
                             columns=ids).T
        list_preds.append(preds)

    preds = pd.concat(list_preds)
    preds = preds.reset_index()
    preds.columns = ['id'] + preds.columns.tolist()[1:]
    
    preds.to_csv(
            os.path.join(EXTERNAL_PATH, "tf_weights_%s.csv" % horizon), index=False
        )

if __name__ == "__main__":
    fire.Fire(train_and_pred)