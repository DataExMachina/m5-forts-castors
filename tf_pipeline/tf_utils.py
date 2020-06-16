import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import gc
import shutil
import os
import json

# imports we know we'll need only for BGS
import skopt
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt.callbacks import CheckpointSaver, VerboseCallback
from skopt import load

from conf import cat_feats, REFINED_PATH, useless_cols, MODELS_PATH

tfkl = tfk.layers
K = tfk.backend

np.random.seed(42)


def df_to_tf(df, cat_feats, useless_cols, use_validation=True):
    '''
    take a dataframe and turn it into the needed objects for tensorflow

    :param df: preprocessed dataframe
    :param cat_feats: categorical features list
    :param useless_cols: useless_cols in the dataframe
    :param use_validation: use a validation set or not
    :return: dictionnary for training (optionnal testing) cardinality list and weight for training

    '''
    df.dropna(inplace=True)
    gc.collect()
    num_feats = df.columns[~df.columns.isin(useless_cols + cat_feats)].to_list()
    train_cols = num_feats + cat_feats

    X_train = df[train_cols]
    y_train = df["sales"]

    X_train[cat_feats] = X_train[cat_feats].astype(np.int32)
    X_train[num_feats] = X_train[num_feats].astype(np.float32)

    if use_validation:
        fake_valid_inds = np.random.choice(
            X_train.index.values, 2_000_000, replace=False)
        train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)

        X_test, y_test = X_train.loc[fake_valid_inds], y_train.loc[fake_valid_inds]
        X_train, y_train = X_train.loc[train_inds], y_train.loc[train_inds]

        X_test[cat_feats] = X_test[cat_feats].astype(np.int32)
        X_test[num_feats] = X_test[num_feats].astype(np.float32)

    cardinality = df[cat_feats].nunique()

    weights = df['d'].str[2:].astype(int)
    weights = weights / np.sum(weights)
    weights_train = weights.loc[X_train.index]

    input_dict = {f"input_{col}": X_train[col] for col in X_train.columns}
    if use_validation:
        input_dict_test = {f"input_{col}": X_test[col] for col in X_train.columns}
        return input_dict, y_train, input_dict_test, y_test, cardinality, weights_train
    else:
        return input_dict, y_train, cardinality, weights_train


# loss functions
def poisson(y_true, y_pred):
    '''
    Loss computed as a Poisson regression
    '''
    return K.mean(K.maximum(.0, y_pred) - y_true * K.log(K.maximum(.0, y_pred) + K.epsilon()), axis=-1)


def tweedie_loss(y_true, y_pred):
    '''
    Tweedie regression, same style as poisson but ... well ... different

    '''
    p = 1.5
    dev = K.pow(y_true, 2 - p) / ((1 - p) * (2 - p)) \
          - y_true * K.pow(K.maximum(.0, y_pred) + K.epsilon(), 1 - p) / (1 - p) \
          + K.pow(K.maximum(.0, y_pred) + K.epsilon(), 2 - p) / (2 - p)

    return K.mean(dev, axis=-1)


alpha = .5


def weighted_loss(y_true, y_pred):
    '''
    make a comprised loss of poisson and tweedie distribution
    '''
    return (1 - alpha) * poisson(y_true, y_pred) + alpha * tweedie_loss(y_true, y_pred)


# function to generate the MLP
def create_mlp(layers_list=None, emb_dim=30, loss_fn='poisson', learning_rate=1e-3, optimizer=tfk.optimizers.Adam,
               cat_feats=cat_feats, num_feats=[], cardinality=[], verbose=0):
    '''
    description :
    generate regression mlp with
    both embedding entries for categorical features and
    standard inputs for numerical features

    params:
    layers_list : list of layers dimensions
    emb_dim : maximum embedding size
    output :
    uncompiled keras model
    '''

    # define our MLP network
    if layers_list is None:
        layers_list = [128, 128, 64, 64, 32]

    layers = []
    output_num = []
    inputs = []
    output_cat = []
    output_num = []

    # sequencial inputs

    # numerical data part
    if len(num_feats) > 1:
        for num_var in num_feats:
            input_num = tfkl.Input(
                shape=(1,), name='input_{0}'.format(num_var))
            inputs.append(input_num)
            output_num.append(input_num)
        output_num = tfkl.Concatenate(name='concatenate_num')(output_num)
        output_num = tfkl.BatchNormalization()(output_num)  # to avoid preprocessing

    else:
        input_num = tfkl.Input(
            shape=(1,), name='input_{0}'.format(num_feats[0]))
        inputs.append(input_num)
        output_num = input_num

    # categorical data input
    for categorical_var in cat_feats:
        no_of_unique_cat = cardinality[categorical_var]
        if verbose == 1:
            print(categorical_var, no_of_unique_cat)
        embedding_size = min(np.ceil(no_of_unique_cat / 2), emb_dim)
        embedding_size = int(embedding_size)
        vocab = no_of_unique_cat + 1

        # functionnal loop
        input_cat = tfkl.Input(
            shape=(1,), name='input_{0}'.format(categorical_var))
        inputs.append(input_cat)
        embedding = tfkl.Embedding(vocab,
                                   embedding_size,
                                   embeddings_regularizer=tf.keras.regularizers.l1(1e-8),
                                   name='embedding_{0}'.format(categorical_var))(input_cat)
        embedding = tfkl.Dropout(0.15)(embedding)
        vec = tfkl.Flatten(name='flatten_{0}'.format(
            categorical_var))(embedding)
        output_cat.append(vec)
    output_cat = tfkl.Concatenate(name='concatenate_cat')(output_cat)

    # concatenate numerical input and embedding output
    dense = tfkl.Concatenate(name='concatenate_all')([output_num, output_cat])

    # dense network
    for i in range(len(layers_list)):
        dense = tfkl.Dense(layers_list[i],
                           name='Dense_{0}'.format(str(i)),
                           activation='elu')(dense)
        dense = tfkl.Dropout(.15)(dense)
        dense = tfkl.BatchNormalization()(dense)

    dense2 = tfkl.Dense(1, name='Output', activation='elu')(dense)
    model = tfk.Model(inputs, dense2)

    opt = optimizer(learning_rate)

    # choose the type of regression
    if loss_fn == 'poisson':
        model.compile(loss=poisson, optimizer=opt, metrics=[
            tf.keras.metrics.RootMeanSquaredError()])
    elif loss_fn == 'tweedie':
        model.compile(loss=tweedie_loss, optimizer=opt, metrics=[
            tf.keras.metrics.RootMeanSquaredError()])
    else:
        raise ValueError(
            "Loss function should be either Poisson or tweedie for now")
    return model


def train_mlp(horizon="validation", task="volume", mlp_params=None, training_params=None):
    df = pd.read_parquet(
        os.path.join(REFINED_PATH, "%s_%s_fe.parquet" % (horizon, task))
    )
    df[cat_feats].fillna(-1,inplace=True)
    df.dropna(inplace=True)
    
    num_feats = df.columns[~df.columns.isin(useless_cols + cat_feats)].to_list()

    input_dict, y_train, cardinality, weights_train = df_to_tf(df, cat_feats, useless_cols, use_validation=False)
    if mlp_params is None:
        mlp_params = {
            'layers_list': [128, 128, 64, 64, 32],  # [512, 256, 128, 64]
            'emb_dim': 30,
            'loss_fn': 'poisson',
            'learning_rate': 1e-3,
            'optimizer': tfk.optimizers.Adam,
            'cat_feats': cat_feats,
            'num_feats': num_feats,
            'cardinality': cardinality,
            'verbose': 1
        }

    mdl = create_mlp(**mlp_params)
    try:
        tfk.utils.plot_model(mdl,show_shapes=True)
    except:
        print('Impossible to plot the model')

              
              
    # checkpointsthe model to reload the best parameters
    model_save = tfk.callbacks.ModelCheckpoint('model_checkpoints',
                                               verbose=0)

    # Early stopping callback
    early_stopping = tfk.callbacks.EarlyStopping('val_root_mean_squared_error',
                                                 patience=10,
                                                 verbose=0,
                                                 restore_best_weights=True)

    if training_params is None:
        training_params = {
            'x': input_dict,
            'y': y_train.values,
            'batch_size': 4096,
            'epochs': 100,
            'shuffle': True,
            'sample_weight': weights_train.values,
            'callbacks': [model_save, early_stopping]
        }

    history = mdl.fit(**training_params)

    mdl.save((os.path.join(MODELS_PATH, "%s_%s_tf.h5" % (horizon, task))))


def grid_train(horizon="validation", task="volume"):
    dim_learning_rate = Real(low=1e-3, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_num_epoch = Integer(low=5, high=150, name='num_epoch')
              
    dim_num_dense_layers = Integer(low=1, high=3, name='num_dense_layers')
    dim_num_dense_nodes = Integer(low=32, high=512, name='num_dense_nodes')
    dim_batch_size = Integer(low=2048, high=10000, name='batch_size')
    dim_emb_dim = Integer(low=10, high=50, name='emb_dim')
    dim_loss_fn = Categorical(categories=['poisson', 'tweedie'], name='loss_fn')
    dim_weigth =  Categorical(categories=[True, False], name='do_weigth')

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
    df[cat_feats].fillna(-1,inplace=True)
    df.dropna(inplace=True)
    
    num_feats = df.columns[~df.columns.isin(useless_cols + cat_feats)].to_list()

    input_dict, y_train, input_dict_test, y_test, cardinality, weights_train = df_to_tf(df, cat_feats, useless_cols,
                                                                                        use_validation=True)

    @use_named_args(dimensions=dimensions)
    def fitness(learning_rate,
                epoch = num_epoch,
                num_dense_layers,
                num_dense_nodes,
                batch_size,
                emb_dim,
                loss_fn,
                dim_weigth,
                ):

        # generate a list of the MLP architecture
        # Enforce a funnel like structure
        list_layer = [num_dense_nodes // (2 ** x) for x in range(num_dense_layers)]

        model = create_mlp(layers_list=list_layer, emb_dim=emb_dim, loss_fn=loss_fn, learning_rate=learning_rate,
                           optimizer=tfk.optimizers.Adam, cat_feats=cat_feats, num_feats=num_feats,
                           cardinality=cardinality, verbose=0)

        print(f'Generated a model with {model.count_params()} trainable parameters')

        try:
            shutil.rmtree('./model_checkpoints')
            print('Old checkpoint remove')
        except:
            pass

        model_save = tfk.callbacks.ModelCheckpoint('model_checkpoints', verbose=0)

        # Early stopping callback
        early_stopping = tfk.callbacks.EarlyStopping('val_root_mean_squared_error',
                                                     patience=10,
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
                verbose=0,
                ) 
              
        else:
            history = model.fit(input_dict,
                                y_train.values,
                                validation_data=(input_dict_test, y_test.values),
                                batch_size=batch_size,
                                epochs=num_epoch,
                                shuffle=True,
                                callbacks=[model_save, early_stopping],
                                verbose=0,
                                )
              
              
              
           
        # return the validation accuracy for the last epoch.
        rmse =  model.evaluate(input_dict_test, y_test.values, batch_size=10000)[-1]

        # Print the classification accuracy.
        print()
        print("RMSE: {:.2}".format(rmse))
        print()

        global best_rmse
        if rmse < best_rmse:
            model.save((os.path.join(MODELS_PATH, "%s_%s_best_tf.h5" % (horizon, task))))
            best_rmse = rmse
            mlp_params = {
                'layers_list': list_layer,  # [512, 256, 128, 64]
                'emb_dim': emb_dim,
                'loss_fn': loss_fn,
                'learning_rate': learning_rate,
                'num_epoch'  : num_epoch,
                'optimizer': tfk.optimizers.Adam,
                'cat_feats': cat_feats,
                'num_feats': num_feats,
                'cardinality': cardinality,
                'verbose': 1
            }
            with open(MODELS_PATH + 'best_params.json', 'w') as fp:
                json.dump(mlp_params, fp)

        print("Best RMSE: {:.2}".format(rmse))

        # Delete the Keras model with these hyper-parameters from memory(not sure if needed in tf2)
        del model, early_stopping, model_save
        # Clear the Keras session
        K.clear_session()
        gc.collect()
        return rmse

    checkpoint_callback = skopt.callbacks.CheckpointSaver("./checkpoint.pkl")
    try:
        res = load('./checkpoint.pkl')
        x0 = res.x_iters
        y0 = res.func_vals
        best_rmse = min(y0)
        gp_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                n_calls=10,
                                x0=x0,              
                                y0=y0,              
                                n_jobs=1,
                                verbose=True,
                                callback=[checkpoint_callback],
                                )
    except:

        best_rmse = 1000
        gp_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                n_calls=20,
                                n_jobs=1,
                                verbose=True,
                                callback=[checkpoint_callback],
                                )
    print(gp_result.best_params_)
