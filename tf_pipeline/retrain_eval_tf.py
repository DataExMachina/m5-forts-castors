from tf_utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import datetime 

# imports we know we'll need only for BGS
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt.callbacks import CheckpointSaver
from skopt import load
import pickle
import os
import tensorflow as tf
import tensorflow.keras as tfk
from pprint import pprint 


CAL_DTYPES={"event_name_1": "category",
            "event_name_2": "category",
            "event_type_1": "category", 
            "event_type_2": "category",
            "weekday": "category", 
            'wm_yr_wk': 'int16', "wday": "int16",
            "month": "int16", "year": "int16",
            "snap_CA": "float32",
            'snap_TX': 'float32',
            'snap_WI': 'float32' }

PRICE_DTYPES = {"store_id": "category",
                "item_id": "category",
                "wm_yr_wk": "int16", 
                "sell_price":"float32" }

cat_feats = ['item_id', 
             'dept_id',
             'store_id',
             'cat_id',
             'state_id'] +\
            ["event_name_1",
             "event_name_2",
             "event_type_1",
             "event_type_2"]


horizon="evaluation"
task="volume"

useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
h = 28 
max_lags = 57
tr_last = 1913+28


def create_dt(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv("./data/raw/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("./data/raw/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("./data/raw/sales_train_evaluation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)

    print('>>> control create df ')
    
    return dt

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


FIRST_DAY = 1 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !
df = create_dt(is_train=True, first_day= FIRST_DAY)
create_fea(df)
print(f"MEMORY USAGE : {df.memory_usage().sum()/1e9}")
df.dropna(inplace=True)
# get the weights for the training (the older the sample the less it will have impact )

num_feats = df.columns[~df.columns.isin(useless_cols+cat_feats)].to_list()
train_cols = num_feats+cat_feats

X_train = df[train_cols]
y_train = df["sales"]


cardinality  = df[cat_feats].max()
print(cardinality)
#weights_train =  weights.loc[X_train.index]

input_dict = {f"input_{col}": X_train[col] for col in X_train.columns}

del df,X_train
gc.collect()

model = tfk.models.load_model(os.path.join(MODELS_PATH, "%s_%s_tf.h5" % ('validation', task)),custom_objects={ 'poisson': poisson })
sgd = tfk.optimizers.SGD(lr=0.0001) # very high for testing
model.optimizer = sgd   #change optimizer

history = model.fit(input_dict,
                    y_train.values,
                    batch_size=4096,
                    epochs=5,
                    shuffle=True,
                    #sample_weight=weights_train.values,
                    #callbacks=[model_save, early_stopping],
                    verbose=1,
                    )


model.save((os.path.join(MODELS_PATH, "%s_%s_tf.h5" % (horizon, task))))
