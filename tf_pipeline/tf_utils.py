import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import gc
import pickle

from conf import cat_feats, REFINED_PATH, useless_cols, MODELS_PATH

tfkl = tfk.layers
K = tfk.backend
np.random.seed(42)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

## evaluation metric

def df_to_tf(df, cat_feats, useless_cols, use_validation=True):
    """
    take a dataframe and turn it into the needed objects for tensorflow

    :param df: preprocessed dataframe
    :param cat_feats: categorical features list
    :param useless_cols: useless_cols in the dataframe
    :param use_validation: use a validation set or not
    :return: dictionnary for training (optionnal testing) cardinality list and weight for training

    """
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
    """
    Loss computed as a Poisson regression
    """
    return K.mean(K.maximum(.0, y_pred) - y_true * K.log(K.maximum(.0, y_pred) + K.epsilon()), axis=-1)


def tweedie_loss(y_true, y_pred):
    """
    Tweedie regression, same style as poisson but ... well ... different

    """
    p = 1.5
    dev = K.pow(y_true, 2 - p) / ((1 - p) * (2 - p)) \
          - y_true * K.pow(K.maximum(.0, y_pred) + K.epsilon(), 1 - p) / (1 - p) \
          + K.pow(K.maximum(.0, y_pred) + K.epsilon(), 2 - p) / (2 - p)

    return K.mean(dev, axis=-1)


alpha = .5


def weighted_loss(y_true, y_pred):
    """
    make a comprised loss of poisson and tweedie distribution
    """
    return (1 - alpha) * poisson(y_true, y_pred) + alpha * tweedie_loss(y_true, y_pred)


# function to generate the MLP
def create_mlp(layers_list=None, emb_dim=30, loss_fn='poisson', learning_rate=1e-3, optimizer=tfk.optimizers.Adam,
               cat_feats=cat_feats, num_feats=None, cardinality=None, verbose=0):
    """
    description :
    generate regression mlp with
    both embedding entries for categorical features and
    standard inputs for numerical features

    params:
    layers_list : list of layers dimensions
    emb_dim : maximum embedding size
    output :
    uncompiled keras model
    """

    # define our MLP network
    if cardinality is None:
        cardinality = []
    if num_feats is None:
        num_feats = []
    if layers_list is None:
        layers_list = [128, 128, 64, 64, 32]

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


def train_mlp(horizon="validation",task='volume', training_params=None):
    df = pd.read_parquet(
        os.path.join(REFINED_PATH, "%s_%s_fe.parquet" % (horizon, task))
    )
    df[cat_feats].fillna(-1, inplace=True)
    df.dropna(inplace=True)

    num_feats = df.columns[~df.columns.isin(useless_cols + cat_feats)].to_list()
    input_dict, y_train, cardinality, weights_train = df_to_tf(df, cat_feats, useless_cols, use_validation=False)
    with open(MODELS_PATH + 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)

    mdl = create_mlp(layers_list=params['layers_list'], emb_dim=params['emb_dim'], loss_fn=params['loss_fn'], learning_rate=params['learning_rate'],
                       optimizer=tfk.optimizers.Adam, cat_feats=cat_feats, num_feats=num_feats,
                       cardinality=params['cardinality'], verbose=0)
    try:
        tfk.utils.plot_model(mdl, show_shapes=True)
    except:
        print('Impossible to plot the model')

    # checkpointsthe model to reload the best parameters
    model_save = tfk.callbacks.ModelCheckpoint('model_checkpoints',
                                               verbose=0)

    # Early stopping callback
    early_stopping = tfk.callbacks.EarlyStopping('root_mean_squared_error',
                                                 patience=10,
                                                 verbose=0,
                                                 restore_best_weights=True)

    if training_params is None:
        if params['do_weigth']:
            training_params = {
                'x': input_dict,
                'y': y_train.values,
                'batch_size': 4096,
                'epochs': params['num_epoch'],
                'shuffle': True,
                'sample_weight': weights_train,
                'callbacks': [model_save, early_stopping]
            }
        else:
            training_params = {
                'x': input_dict,
                'y': y_train.values,
                'batch_size': 4096,
                'epochs': params['num_epoch'],
                'shuffle': True,
                'callbacks': [model_save, early_stopping]
            }

    mdl.fit(**training_params)

    mdl.save((os.path.join(MODELS_PATH, "%s_%s_tf.h5" % (horizon, task))))
