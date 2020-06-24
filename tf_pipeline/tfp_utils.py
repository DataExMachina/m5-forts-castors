'''
Optionnal part to run probabilistic deep learning 

'''
from conf import * 
from tf_utils import * 
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import tensorflow_probability as tfp



tfd = tfp.distributions

# prior distribution on the weights 
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

# layer posterior distribution with mean field approximation  
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + 0.02*tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

# probabilistic  loss functions
def neg_log_likelihood_continuous(y_true, y_pred):
    '''
    negative log likelyhood for stricly positive distributions 
    
    '''
    return -y_pred.prob(y_true+1e-6)


def neg_log_likelihood_discrete(y_true, y_pred):
    '''
    negloglik when we don't care about the probability in zero 
    '''
    return -y_pred.log_prob(y_true)

# custom metrics


def rmse(y_true, y_pred):
    '''
    compact implemtation of the rmse 
    '''
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred-y_true)))


# to much scaling: high variance, no enough -> converge to the likelyhood
kl_weight = batch_size/training_size 
kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                          kl_weight)  # KL over the batch of 2048 (should normalize by the number of mini batch)


def create_mlp(layers_list=[512, 512, 512, 64],
               type_output='poisson'):
    '''
    description : 
    generate regression mlp with
    both embedding entries for categorical features and 
    standard inputs for numerical features

    params:
    layers_list : list of layers dimensions 
    output :
    compiled keras model  
    '''

    # define our MLP network
    layers = []
    output_num = []
    inputs = []
    output_cat = []
    output_num = []

    # numerical data part
    if len(num_feats) > 1:
        for num_var in num_feats:
            input_num = tfkl.Input(
                shape=(1,), name='input_{0}'.format(num_var))
            inputs.append(input_num)
            output_num.append(input_num)
        output_num = tfkl.Concatenate(name='concatenate_num')(output_num)
        #output_num = tfkl.BatchNormalization()(output_num)

    else:
        input_num = tfkl.Input(
            shape=(1,), name='input_{0}'.format(numeric_features[0]))
        inputs.append(input_num)
        output_num = input_num

    # create an embedding for every categorical feature
    for categorical_var in cat_feats:
        # should me nunique() but events are poorly preprocessed
        no_of_unique_cat = cardinality[categorical_var]
        print(categorical_var, no_of_unique_cat)
        embedding_size = min(np.ceil((no_of_unique_cat)/2), 30)
        embedding_size = int(embedding_size)
        vocab = no_of_unique_cat+1
        # functionnal loop
        input_cat = tfkl.Input(
            shape=(1,), name='input_{0}'.format(categorical_var))
        inputs.append(input_cat)
        embedding = tfkl.Embedding(vocab,
                                   embedding_size,
                                   embeddings_regularizer = tf.keras.regularizers.l1(1e-8),
                                   name='embedding_{0}'.format(categorical_var))(input_cat)
        embedding = tfkl.Dropout(0.1)(embedding)
        vec = tfkl.Flatten(name='flatten_{0}'.format(
            categorical_var))(embedding)
        output_cat.append(vec)
    output_cat = tfkl.Concatenate(name='concatenate_cat')(output_cat)

    # concatenate numerical input and embedding output
    dense = tfkl.Concatenate(name='concatenate_all')([output_num, output_cat])

    for i in range(len(layers_list)):
        dense = tfkl.Dense(layers_list[i],
                           name='Dense_{0}'.format(str(i)),
                           activation='elu')(dense)
        dense = tfkl.Dropout(.1)(dense)
        dense = tfkl.BatchNormalization()(dense)

    # lognormal
    if type_output == 'gaussian':
        dense2 = tfk.layers.Dense(2,
                                  activation='softplus',
                                  name='Output'
                                  )(dense)

        output = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(t[..., 0], scale=t[..., 1:]))(dense2)

    # Poisson
    elif type_output == 'poisson':
        dense2 = tfk.layers.Dense(1,
                                  name='Output')(dense)
        output = tfp.layers.DistributionLambda(
            lambda t: tfd.Poisson(rate=tf.math.softplus(t[..., 0])))(dense2)

    # Gamma
    elif type_output == 'gamma':
        dense2 = tfk.layers.Dense(2,
                                  name='Output',
                                  activation='softplus',
                                  )(dense)

        output = tfp.layers.DistributionLambda(
            lambda t: tfd.Gamma(concentration=.01*t[..., 0],
                                rate=.01*t[..., 1]))(dense2)
    else:
        output = tfk.Dense(1)

    model = tfk.Model(inputs, output)
    opt = tfk.optimizers.Nadam(learning_rate=1e-2)

    model.compile(loss=neg_log_likelihood_continuous,
                  optimizer=opt,
                  metrics=[rmse])
    return model


try:
    del mdl_tfp
    print('model deleted ')
except:
    pass


def create_mlp_full_bayes(layers_list=[512, 256, 128, 64],
                          type_output='poisson'):
    '''
    description : 
    generate regression mlp with
    both embedding entries for categorical features and 
    standard inputs for numerical features

    params:
    layers_list : list of layers dimensions 
    output :
    compiled keras model  
    '''

    # define our MLP network
    layers = []
    output_num = []
    inputs = []
    output_cat = []
    output_num = []

    # numerical data part
    if len(num_feats) > 1:
        for num_var in num_feats:
            input_num = tfkl.Input(
                shape=(1,), name='input_{0}'.format(num_var))
            inputs.append(input_num)
            output_num.append(input_num)
        output_num = tfkl.Concatenate(name='concatenate_num')(output_num)
        #output_num = tfkl.BatchNormalization()(output_num)

    else:
        input_num = tfkl.Input(
            shape=(1,), name='input_{0}'.format(numeric_features[0]))
        inputs.append(input_num)
        output_num = input_num

    # create an embedding for every categorical feature
    for categorical_var in cat_feats:
        # should me nunique() but events are poorly preprocessed
        no_of_unique_cat = cardinality[categorical_var]
        print(categorical_var, no_of_unique_cat)
        embedding_size = 10  # min(np.ceil((no_of_unique_cat)/2), 10)
        embedding_size = int(embedding_size)
        vocab = no_of_unique_cat+1
        # functionnal loop
        input_cat = tfkl.Input(
            shape=(1,), name='input_{0}'.format(categorical_var))
        inputs.append(input_cat)
        embedding = tfkl.Embedding(
            vocab, embedding_size, name='embedding_{0}'.format(
                categorical_var),
            embeddings_initializer='random_uniform'
        )(input_cat)
        vec = tfkl.Flatten(name='flatten_{0}'.format(
            categorical_var))(embedding)

        output_cat.append(vec)
    output_cat = tfkl.Concatenate(name='concatenate_cat')(output_cat)

    # concatenate numerical input and embedding output
    dense = tfkl.Concatenate(name='concatenate_all')([output_num, output_cat])

    for i in range(len(layers_list)):
        dense = tfp.layers.DenseVariational(layers_list[i],
                                            activation='elu',
                                            name='Dense_{0}'.format(str(i)),
                                            make_posterior_fn=posterior_mean_field,
                                            make_prior_fn=prior_trainable,
                                            kl_weight=1/2048,
                                            )(dense)
        dense = tfp.bijectors.BatchNormalization()(dense)

    if type_output == 'gaussian':
        dense2 = tfp.layers.DenseVariational(2,
                                             #activation = 'softplus',
                                             make_posterior_fn=posterior_mean_field,
                                             make_prior_fn=prior_trainable,
                                             kl_weight=1/2048,
                                             )(dense)
        output = tfp.layers.DistributionLambda(
            lambda t: tfd.LogNormal(tf.math.softplus(0.05 * t[..., 0]),
                                    scale=1e-6 + tf.math.softplus(0.05 * t[..., 1:])))(dense2)
    # Poisson
    elif type_output == 'poisson':
        dense2 = tfp.layers.DenseVariational(2,
                                             posterior_mean_field,
                                             prior_trainable,
                                             # kl_weight=1/100000
                                             )(dense)
        dense2 = tfp.layers.DenseFlipout(2)(dense)
        output = tfp.layers.DistributionLambda(
            lambda t: tfd.Poisson(rate=tf.math.softplus(t[..., 0])))(dense2)

    # Gamma
    elif type_output == 'gamma':
        #         dense2 = tfp.layers.DenseFlipout(2,
        #                                   name='Output',
        #                                   activation = 'softplus',
        # #                                   kernel_prior =prior_trainable,
        # #                                  kernel_divergence_fn=kl_divergence_function,
        #                                  )(dense)
        dense2 = tfp.layers.DenseVariational(2,
                                             name='Output',
                                             activation='linear',
                                             make_posterior_fn=posterior_mean_field,
                                             kl_weight=1/2048,

                                             make_prior_fn=prior_trainable,
                                             )(dense)
        output = tfp.layers.DistributionLambda(
            lambda t: tfd.Gamma(concentration=t[..., 0],
                                rate=t[..., 1]))(dense2)

    else:
        output = tfk.Dense(1, kernel_divergence_fn=kl_divergence_function,
                           name='Output')(dense)

    model = tfk.Model(inputs, output)
    opt = tfk.optimizers.Adam(learning_rate=1e-3)
    # kl divergence is direclty added to the loss function
    model.compile(loss=neg_log_likelihood_continuous, optimizer=opt,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model