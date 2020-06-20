from tf_utils import *
import seaborn as sns
import matplotlib.pyplot as plt

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

dim_learning_rate = Real(low=1e-3, high=1e-2, prior='log-uniform', name='learning_rate')
dim_num_epoch = Integer(low=3, high=100, name='num_epoch')

dim_num_dense_layers = Integer(low=1, high=6, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=32, high=512, name='num_dense_nodes')
dim_batch_size = Integer(low=2048, high=10000, name='batch_size')
dim_emb_dim = Integer(low=10, high=50, name='emb_dim')
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
df = pd.read_parquet(
    os.path.join(REFINED_PATH, "%s_%s_fe.parquet" % (horizon, task))
)
df[cat_feats].fillna(-1, inplace=True)
df.dropna(inplace=True)

num_feats = df.columns[~df.columns.isin(useless_cols + cat_feats)].to_list()

input_dict, y_train, input_dict_test, y_test, cardinality, weights_train = df_to_tf(df, cat_feats, useless_cols,
                                                                                    use_validation=True)

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

    # try:
    #     shutil.rmtree('./model_checkpoints')
    #     print('Old checkpoint remove')
    # except:
    #     pass

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
                'do_weigth' : do_weigth
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
                            n_calls=20,
                            verbose=10,
                            callback=[checkpoint_callback],
                            )
print('Best results obtained : ')
print(gp_result.x)