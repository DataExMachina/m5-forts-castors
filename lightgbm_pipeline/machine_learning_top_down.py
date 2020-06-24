import os
from datetime import datetime, timedelta
import numpy as np

import lightgbm as lgb
import pandas as pd
import fire

RAW_PATH = "data/raw/"
REFINED_PATH = "data/refined/"
EXTERNAL_PATH = "data/external/"
SUBMIT_PATH = "data/submission/"
MODELS_PATH = "data/models/"

CAL_DTYPES = {
    "event_name_1": "category",
    "event_name_2": "category",
    "event_type_1": "category",
    "event_type_2": "category",
    "weekday": "category",
    "wm_yr_wk": "int16",
    "wday": "int16",
    "month": "int16",
    "year": "int16",
    "snap_CA": "float32",
    "snap_TX": "float32",
    "snap_WI": "float32",
}

PRICE_DTYPES = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "sell_price": "float32",
}


def train(horizon="validation"):

    df = pd.read_parquet(
        os.path.join(REFINED_PATH, "%s_%s_fe.parquet" % (horizon, "volume"))
    )
    cat_feats = ["dept_id", "store_id", "cat_id", "state_id"]
    df = df.groupby(cat_feats+['date'])['lag_7', 'lag_28', 'rmean_7_7', 'rmean_28_7', 'rmean_7_28', 'rmean_28_28', 'sales'].sum().reset_index()
    train_cols = cat_feats + ['lag_7', 'lag_28', 'rmean_7_7', 'rmean_28_7', 'rmean_7_28', 'rmean_28_28']
    X_train = df[train_cols]
    y_train = df["sales"]

    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_feats, free_raw_data=False
    )

    params = {
        "metric": "rmse",
        "force_row_wise": True,
        "learning_rate": 0.075,
        "sub_feature": 0.8,
        "sub_row": 0.75,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "nthread": 16,
        "metric": ["rmse"],
        "verbosity": 1,
    }

    params["objective"] = "poisson"
    params["num_iterations"] = 2500

    m_lgb = lgb.train(params, train_data)
    m_lgb.save_model(os.path.join(MODELS_PATH, "%s_lgb_top_down.txt" % (horizon)))


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
        if col=='store_id' or col=='dept_id':
            dt['copy_%s' % col] = dt[col]
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
        elif col != "id":
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
        id_vars=catcols+['copy_dept_id', 'copy_store_id'],
        value_vars=[col for col in dt.columns if col.startswith("d_")],
        var_name="d",
        value_name="sales",
    )

    dt = dt.merge(cal, on="d", copy=False)
    dt = dt.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
    dt = dt.merge(rates, how="left")

    return dt


def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["store_id", 'dept_id', "sales"]].groupby(["store_id", 'dept_id'])["sales"].shift(lag)

    wins = [7, 28]
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = (
                dt[["store_id", 'dept_id', lag_col]]
                .groupby(["store_id", 'dept_id'])[lag_col]
                .transform(lambda x: x.rolling(win).mean())
            )

    return dt

def predict(horizon="validation"):
    if horizon == "validation":
        tr_last = 1913
        fday = datetime(2016, 4, 25)
    elif horizon == "evaluation":
        tr_last = 1941
        fday = datetime(2016, 4, 25) + timedelta(days=28)
    else:
        raise ValueError("Wrong value for horizon arg.")

    dataframe = create_dt(horizon, tr_last)
    cat_mapping = dataframe[['dept_id', 'store_id', 'copy_dept_id', 'copy_store_id']].drop_duplicates()

    m_lgb = lgb.Booster(
        model_file=os.path.join(MODELS_PATH, "%s_lgb_top_down.txt" % (horizon))
    )

    cat_feats = ["dept_id", "store_id", "cat_id", "state_id"]

    dataframe = dataframe.groupby(cat_feats+['date'])\
                         .agg(sales=("sales", 'sum'), rate=("rate", "mean"))\
                         .reset_index()
    train_cols = cat_feats + ['lag_7', 'lag_28', 'rmean_7_7', 'rmean_28_7', 'rmean_7_28', 'rmean_28_28']
    
    for i in range(0, 28):
        day = fday + timedelta(days=i)
        tst = dataframe[
            (dataframe.date >= day - timedelta(days=366)) & (dataframe.date <= day)
        ].copy()

        tst = create_fea(tst)
        tst = tst.loc[tst.date == day, train_cols + ["rate"]]
        
        dataframe.loc[dataframe.date == day, "sales"] = tst["rate"] * m_lgb.predict(
            tst[train_cols]
            )

    te_sub = dataframe.loc[dataframe.date >= fday, ["store_id", "dept_id", "sales"]].copy()
    te_sub = te_sub.merge(cat_mapping).drop(['dept_id', 'store_id'], axis=1).rename(columns={'copy_dept_id': 'dept_id', 'copy_store_id': 'store_id'})

    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby(["store_id", "dept_id"])["store_id", "dept_id"].cumcount() + 1]
    te_sub = (
        te_sub.set_index(["store_id", "dept_id", "F"])
        .unstack()["sales"][[f"F{i}" for i in range(1, 29)]]
        .reset_index()
    )
    #te_sub.fillna(0.0, inplace=True)
    
    te_sub.to_csv(
        os.path.join(EXTERNAL_PATH, "lgb_estim_top_down_%s.csv" % horizon), index=False
        )

def ml_pipeline(horizon="validation", ml="train_and_predict"):
    if ml == "train_and_predict":
        train(horizon)
        predict(horizon)
    elif ml == "predict":
        predict(horizon)
    else:
        raise ValueError('ml arg must be "train_and_predict" or "predict"')


if __name__ == "__main__":
    fire.Fire(ml_pipeline)
