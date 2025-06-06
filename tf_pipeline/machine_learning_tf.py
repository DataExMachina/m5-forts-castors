import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from datetime import timedelta, datetime
from tqdm import tqdm
import fire
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
from tf_utils import train_mlp
from conf import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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


def predict(horizon="validation", task="volume"):
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

    mdl = tfk.models.load_model((os.path.join(MODELS_PATH, "%s_%s_full_best_tf.h5" % (horizon, task))))

    for i in tqdm(range(0, 28)):
        day = fday + timedelta(days=i)
        tst = dataframe[
            (dataframe.date >= day - timedelta(days=366)) & (dataframe.date <= day)
            ].copy()

        tst = create_fea(tst)
        train_cols = tst.columns[~tst.columns.isin(useless_cols)]
        tst = tst.loc[tst.date == day, train_cols.tolist() + ["rate"]]

        input_dict_predict = {f"input_{col}": tst[col] for col in train_cols}

        if task == "volume":
            predictions = mdl.predict(input_dict_predict, batch_size=10000)

            dataframe.loc[dataframe.date == day, "sales"] = tst["rate"] * predictions.ravel()

        elif task == "share":
            #
            ## not ready yet for deep learning 
            #
            dataframe.loc[te.date == day, "sales"] = mdl.predict(input_dict_predict, batch_size=10000)
            shares = (
                dataframe.groupby(["dept_id", "store_id", "date"])["sales"]
                    .sum()
                    .reset_index()
                    .rename(columns={"sales": "gp_sales"})
            )
            dataframe = dataframe.merge(shares, how="left")
            dataframe["sales"] = dataframe["sales"] / dt["gp_sales"]
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
        print(te_sub)

    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount() + 1]
    te_sub = (
        te_sub.set_index(["id", "F"])
            .unstack()["sales"][[f"F{i}" for i in range(1, 29)]]
            .reset_index()
    )
    te_sub.fillna(0.0, inplace=True)

    if task == "volume":
        te_sub.to_csv(
            os.path.join(SUBMIT_PATH, "tf_estim_%s.csv" % horizon), index=False
        )
    else:
        te_sub.to_csv(
            os.path.join(EXTERNAL_PATH, "tf_weights_%s.csv" % horizon), index=False
        )


def ml_pipeline(horizon="validation", task="volume", ml="predict"):
    if ml == "train_and_predict":
        train_mlp(horizon, task)
        predict(horizon, task)
    elif ml == "predict":
        predict(horizon, task)
        os.system('shutdown -s')

    else:
        raise ValueError('ml arg must be "train_and_predict" or "predict" or "train_grid"')

if __name__ == "__main__":
    fire.Fire(ml_pipeline)


