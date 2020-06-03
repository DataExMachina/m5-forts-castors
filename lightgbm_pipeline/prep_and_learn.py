from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb

CAL_DTYPES = {
    "event_name_1": "category",
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
    'snap_WI': 'float32' 
}
PRICE_DTYPES = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "sell_price":"float32"
}
