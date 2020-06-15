import os

import fire
import numpy as np
import json
import pandas as pd

from tf_utils import train_mlp, grid_train
from conf import RAW_PATH, PRICE_DTYPES, CAL_DTYPES, REFINED_PATH


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


def ml_pipeline(horizon="validation", task="volume", ml="train_and_predict"):
    if ml == "train_and_predict":
        train_mlp(horizon, task)
        # predict(horizon, task)
    # elif ml == "predict":
        # predict(horizon, task)

    elif ml=='train_grid':
        grid_train(horizon, task)
    else:
        raise ValueError('ml arg must be "train_and_predict" or "predict"')

#
if __name__ == "__main__":
    fire.Fire(ml_pipeline)



