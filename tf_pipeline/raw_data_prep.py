import os
from datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
import fire
from conf import * 

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


def data_prep(horizon="validation"):
    if horizon == "validation":
        tr_last = 1913
    elif horizon == "evaluation":
        tr_last = 1941
    else:
        raise ValueError("Wrong value for horizon arg.")

    dataframe = create_dt(horizon, tr_last)
    dataframe.to_parquet(os.path.join(INTERIM_PATH, "%s_raw.parquet" % horizon))


if __name__ == "__main__":
    fire.Fire(data_prep)
