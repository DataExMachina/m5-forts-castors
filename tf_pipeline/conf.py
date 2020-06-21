RAW_PATH = "../data/raw/"
INTERIM_PATH = "../data/interim/"
REFINED_PATH = "../data/refined/"
EXTERNAL_PATH = "../data/external/"
SUBMIT_PATH = "../data/submission/"
MODELS_PATH = "../data/models/"


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

## features

cat_feats = ['item_id',
             'dept_id',
             'store_id',
             'cat_id',
             'state_id'] + \
            ["event_name_1",
             "event_name_2",
             "event_type_1",
             "event_type_2"]

useless_cols = ["id",
                "date",
                "sales",
                "d",
                "wm_yr_wk",
                "weekday",
                "rate"
               ]



