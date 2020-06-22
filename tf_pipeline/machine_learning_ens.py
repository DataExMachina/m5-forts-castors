import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

from datetime import datetime, timedelta
import fire
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow.keras as tfk
from conf import *
from conf import (
    MODELS_PATH,
    SUBMIT_PATH,
    EXTERNAL_PATH,
    RAW_PATH,
    PRICE_DTYPES,
    CAL_DTYPES,
)
from tf_utils import train_mlp
from tqdm import tqdm
import pickle

# silence tensorflow importing library
logging.getLogger("tensorflow").setLevel(logging.FATAL)


def train(horizon="validation", task="volume"):
    df = pd.read_parquet(
        os.path.join(REFINED_PATH, "%s_%s_fe.parquet" % (horizon, task))
    )
    cat_feats = ["item_id", "dept_id", "store_id", "cat_id", "state_id"] + [
        "event_name_1",
        "event_name_2",
        "event_type_1",
        "event_type_2",
    ]
    useless_cols = ["id", "date", "sales", "d", "wm_yr_wk", "weekday", "rate"]
    train_cols = df.columns[~df.columns.isin(useless_cols)]
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

    if task == "volume":
        params["objective"] = "poisson"
        params["num_iterations"] = 15000
    elif task == "share":
        params["objective"] = "xentropy"
        params["num_iterations"] = 2000

    m_lgb = lgb.train(params, train_data)
    m_lgb.save_model(os.path.join(MODELS_PATH, "%s_%s_lgb.txt" % (horizon, task)))


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


def predict(horizon="validation", task="volume", ensembling_type="avg"):
    if horizon == "validation":
        tr_last = 1913
        fday = datetime(2016, 4, 25)
    elif horizon == "evaluation":
        tr_last = 1941
        fday = datetime(2016, 4, 25) + timedelta(days=28)
    else:
        raise ValueError("Wrong value for horizon arg.")

    dataframe = create_dt(horizon, tr_last)

    if task == "share":
        dataframe = compute_share(dataframe)
    elif task != "volume":
        raise ValueError("Wrong value for task.")

    # gather both models
    print(">>>  load the two models ")
    m_lgb = pickle.load(
        open(os.path.join(MODELS_PATH, "%s_%s_lgb.pickle" % (horizon, task)), "rb")
    )

    print("--- lgbm ok  ")
    m_tf = tfk.models.load_model(
        (os.path.join(MODELS_PATH, "%s_%s_tf.h5" % (horizon, task)))
    )
    print("--- tf ok  ")
    print(">>> start to make predictions ")
    for i in tqdm(range(0, 28)):
        day = fday + timedelta(days=i)
        tst = dataframe[
            (dataframe.date >= day - timedelta(days=366)) & (dataframe.date <= day)
        ].copy()

        tst = create_fea(tst)
        train_cols = tst.columns[~tst.columns.isin(useless_cols)]
        tst = tst.loc[tst.date == day, train_cols.tolist() + ["rate"]]

        if task == "volume":
            input_dict_predict = {f"input_{col}": tst[col] for col in train_cols}
            if ensembling_type == "avg":
                predictions = (
                    tst["rate"]
                    * (
                        m_lgb.predict(tst[train_cols])
                        + m_tf.predict(input_dict_predict, batch_size=10000).flatten()
                    )
                    / 2
                )
            dataframe.loc[dataframe.date == day, "sales"] = predictions
        elif task == "share":
            dataframe.loc[dataframe.date == day, "sales"] = m_lgb.predict(
                tst[train_cols]
            )
            shares = (
                dataframe.groupby(["dept_id", "store_id", "date"])["sales"]
                .sum()
                .reset_index()
                .rename(columns={"sales": "gp_sales"})
            )
            dataframe = dataframe.merge(shares, how="left")
            dataframe["sales"] = dataframe["sales"] / dataframe["gp_sales"]
            dataframe.drop(["gp_sales"], axis=1, inplace=True)

    te_sub = dataframe.loc[dataframe.date >= fday, ["id", "sales"]].copy()
    if horizon == "validation":
        te_sub.loc[dataframe.date >= fday + timedelta(days=28), "id"] = te_sub.loc[
            dataframe.date >= fday + timedelta(days=28), "id"
        ].str.replace("validation", "evaluation")
    else:
        te_sub.loc[dataframe.date >= fday + timedelta(days=28), "id"] = te_sub.loc[
            dataframe.date >= fday + timedelta(days=28), "id"
        ]
        te_sub_validation = te_sub.copy()
        te_sub_validation["id"] = te_sub_validation["id"].str.replace(
            "evaluation", "validation"
        )
        te_sub = pd.concat([te_sub, te_sub_validation])

    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount() + 1]
    te_sub = (
        te_sub.set_index(["id", "F"])
        .unstack()["sales"][[f"F{i}" for i in range(1, 29)]]
        .reset_index()
    )
    te_sub.fillna(0.0, inplace=True)

    if task == "volume":
        te_sub.to_csv(
            os.path.join(SUBMIT_PATH, "ens_estim_%s.csv" % horizon), index=False
        )
    else:
        te_sub.to_csv(
            os.path.join(EXTERNAL_PATH, "ens_weights_%s.csv" % horizon), index=False
        )


def ml_pipeline(horizon="validation", task="volume", ml="predict"):
    if ml == "train_and_predict":
        train(horizon, task)
        train_mlp(horizon, task)
        predict(horizon, task)
    elif ml == "predict":
        predict(horizon, task)
    else:
        raise ValueError('ml arg must be "train_and_predict" or "predict"')


if __name__ == "__main__":
    fire.Fire(ml_pipeline)
