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

print('start grid search')
horizon="validation"
task="volume"

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

useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]

h = 28 
max_lags = 57
tr_last = 1913
fday = datetime.datetime(2016,4, 25) 

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
    dt = pd.read_csv("./data/raw/sales_train_validation.csv", 
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
weights = df['d'].str[2:].astype(int)
weights = weights/np.max(weights)

num_feats = df.columns[~df.columns.isin(useless_cols+cat_feats)].to_list()
train_cols = num_feats+cat_feats

X_train = df[train_cols]
y_train = df["sales"]

np.random.seed(777)
fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)
train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)

X_test,y_test = X_train.loc[fake_valid_inds],y_train.loc[fake_valid_inds]
X_train,y_train = X_train.loc[train_inds],y_train.loc[train_inds]
cardinality  = df[cat_feats].max()
weights_train =  weights.loc[X_train.index]

input_dict = {f"input_{col}": X_train[col] for col in X_train.columns}
input_dict_test = {f"input_{col}": X_test[col] for col in X_train.columns}

del df,X_train,X_test
gc.collect()

dim_learning_rate = Real(low=1e-3, high=1e-2, prior='log-uniform', name='learning_rate')
dim_num_epoch = Integer(low=3, high=100, name='num_epoch')

dim_num_dense_layers = Integer(low=1, high=6, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=32, high=512, name='num_dense_nodes')
dim_batch_size = Integer(low=2048, high=10000, name='batch_size')
dim_emb_dim = Integer(low=30, high=100, name='emb_dim')
dim_loss_fn = Categorical(categories=['poisson', 'tweedie'], name='loss_fn')
dim_weigth = Categorical(categories=[True, False], name='do_weigth')

dimensions = [dim_learning_rate,
              dim_num_epoch,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_emb_dim,
              dim_batch_size,
              dim_loss_fn,
              dim_weigth
              ]


### ORIGINAL DATA PREPROCESSING 

# df = pd.read_parquet(
#     os.path.join(REFINED_PATH, "%s_%s_fe.parquet" % (horizon, task))
# )
# df[cat_feats].fillna(-1, inplace=True)
# df.dropna(inplace=True)

# num_feats = df.columns[~df.columns.isin(useless_cols + cat_feats)].to_list()

# input_dict, y_train, input_dict_test, y_test, cardinality, weights_train = df_to_tf(df, cat_feats, useless_cols,
#                                                                                     use_validation=True)

@use_named_args(dimensions=dimensions)
def fitness(learning_rate,
            num_epoch,
            num_dense_layers,
            num_dense_nodes,
            batch_size,
            emb_dim,
            loss_fn,
            do_weigth,
            ):

    # generate a list of the MLP architecture
    # Enforce a funnel like structure
    list_layer = [num_dense_nodes // (2 ** x) for x in range(num_dense_layers)]

    model = create_mlp(layers_list=list_layer, emb_dim=emb_dim, loss_fn=loss_fn, learning_rate=learning_rate,
                       optimizer=tfk.optimizers.Adam, cat_feats=cat_feats, num_feats=num_feats,
                       cardinality=cardinality, verbose=0)

    print(f'Generated a model with {model.count_params()} trainable parameters')

    model_save = tfk.callbacks.ModelCheckpoint('model_checkpoints', verbose=0)

    # Early stopping callback
    early_stopping = tfk.callbacks.EarlyStopping('val_root_mean_squared_error',
                                                 patience=15,
                                                 verbose=0,
                                                 restore_best_weights=True)

    if do_weigth:
        history = model.fit(input_dict,
                            y_train.values,
                            validation_data=(input_dict_test, y_test.values),
                            batch_size=batch_size,
                            epochs=num_epoch,
                            shuffle=True,
                            sample_weight=weights_train.values,
                            callbacks=[model_save, early_stopping],
                            verbose=2,
                            )

    else:
        history = model.fit(input_dict,
                            y_train.values,
                            validation_data=(input_dict_test, y_test.values),
                            batch_size=batch_size,
                            epochs=num_epoch,
                            shuffle=True,
                            callbacks=[model_save, early_stopping],
                            verbose=2,
                            )

    # return the validation accuracy for the last epoch.
    rmse = model.evaluate(input_dict_test, y_test.values, batch_size=10000)[-1]

    # Print the classification accuracy.
    print()
    print("RMSE: {:.2}".format(rmse))
    print()

    global best_rmse
    if rmse < best_rmse:
        print('--- Better model found ')
        model.save((os.path.join(MODELS_PATH, "%s_%s_best_tf.h5" % (horizon, task))))
        best_rmse = rmse
        mlp_params = {
                'layers_list': list_layer,  # [512, 256, 128, 64]
                'emb_dim': emb_dim,
                'loss_fn': loss_fn,
                'learning_rate': learning_rate,
                'num_epoch': num_epoch,
                'optimizer': 'Adam',
                'cat_feats': cat_feats,
                'num_feats': num_feats,
                'cardinality': cardinality,
                'do_weigth' : do_weigth,
                'batch_size' : batch_size,
                }

        plt.figure(figsize=(20,10))
        res = y_test.values.flatten()-model.predict(input_dict_test,batch_size=10000).flatten()

        sns.distplot(res, bins=1000)
        plt.xlim((-10,10))
        plt.savefig(MODELS_PATH + 'best_error_dist.png')
        plt.close('all')

        with open(MODELS_PATH + 'best_params.pkl', 'wb') as handle:
              pickle.dump(mlp_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Best RMSE: {:.2}".format(best_rmse))

    # Delete the Keras model with these hyper-parameters from memory(not sure if needed in tf2)
    del model, early_stopping, model_save
    # Clear the Keras session
    K.clear_session()
    gc.collect()
    return rmse

checkpoint_callback = CheckpointSaver((os.path.join(MODELS_PATH, "%s_%s_checkpoint.pkl" % (horizon, task))))
global best_rmse
best_rmse=100
try:    
    res = load((os.path.join(MODELS_PATH, "%s_%s_checkpoint.pkl" % (horizon, task))))
    x0 = res.x_iters
    y0 = res.func_vals
    best_rmse = min(y0)
    print('loading gp  weights')
    gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=30,
                            x0=x0,
                            y0=y0,
                            verbose=10,
                            callback=[checkpoint_callback],
                            )
except:
    print('Starting new grid search')
    gp_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=30,
                            verbose=10,
                            callback=[checkpoint_callback],
                            )
print('Best results obtained : ')
print(gp_result.x)