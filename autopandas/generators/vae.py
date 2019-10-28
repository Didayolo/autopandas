# Variational Autoencoder
# Inspired from: http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/

# Imports
from warnings import warn
from .autoencoder import AE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model, Sequential
import autopandas

class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
        to the final model loss.
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs

class VAE(AE):
    def __init__(self, input_dim, layers=[], latent_dim=2, architecture='fully', epsilon_std=1.0, loss='nll', optimizer='rmsprop', decoder_layers=None):
        """ Variational Autoencoder.

            :param input_dim: Input/output size.
            :param layers: Dimension of intermediate layers (encoder and decoder).
                                     It can be:
                                     - an integer (one intermediate layer)
                                     - a list of integers (several intermediate layers)
            :param latent_dim: Dimension of latent space layer.
            :param architecture: 'fully', 'cnn'.
            :param espilon_std: Standard deviation of gaussian distribution prior.
            :param decoder_layers: Dimension of intermediate decoder layers for asymmetrical architectures.
        """
        if isinstance(layers, int): # 1 intermediate layers
            layers = [layers]
        # attributes
        self.input_dim = input_dim
        self.layers = layers
        self.decoder_layers = decoder_layers or layers[::-1]
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        # for data frame indexes
        self.columns = None
        self.indexes = None
        # loss function
        loss_function = self.init_loss(loss=loss)
        # init architecture
        autoencoder, encoder, decoder = self.init_model(architecture=architecture)
        autoencoder.compile(optimizer=optimizer, loss=loss_function)
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder

    def _init_model_fc(self):
        """ Initialize fully connected architecture.
        """
        # encoder architecture
        x = Input(shape=(self.input_dim,))
        h = x
        for layer_dim in self.layers:
            h = Dense(layer_dim, activation='relu')(h)

        # decoder architecture
        latent_input = Input(shape=(self.latent_dim,))
        decoder = latent_input
        for layer_dim in self.decoder_layers:
            decoder = Dense(layer_dim, activation='relu')(decoder)
        decoder = Dense(self.input_dim, activation='sigmoid')(decoder)
        decoder = Model(latent_input, decoder)

        # variational layer
        z_mu = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        # sampling
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
        eps = Input(tensor=K.random_normal(stddev=self.epsilon_std,
                                           shape=(K.shape(x)[0], self.latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])
        x_pred = decoder(z)
        autoencoder = Model([x, eps], x_pred)
        encoder = Model(x, z_mu)
        return autoencoder, encoder, decoder

    def _init_model_cnn(self, kernel=(3, 3), pool=(2, 2), strides=(2, 2), dense_layer=None):
        """ Initialize CNN architecture.

            :param dense_layer: supplementary dense layer (in encoder and decoder) between convolution and latent space.
        """
        warn('strides argument is currently not implemented.')
        if self.layers != self.decoder_layers[::-1]:
            warn('self.layers is {} and self.decoder_layers is {}. Use asymmetric architecture with CNN wisely.'.format(self.layers, self.decoder_layers))
        # encoder architecture
        x = Input(shape=self.input_dim)
        h = x
        for layer_dim in self.layers:
            h = Conv2D(layer_dim, kernel, activation='relu', padding='same')(h)
            h = MaxPooling2D(pool, padding='same')(h)
        new_shape = h.shape[1:]
        h = Flatten()(h)
        flatten_dim = h.shape[1]
        if dense_layer is not None: # add dense layer
            h = Dense(dense_layer)(h)
        # variational layer
        z_mu = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

        # decoder architecture
        latent_input = Input(shape=(self.latent_dim,))
        if dense_layer is not None: # add dense layer
            decoder = Dense(dense_layer)(latent_input)
            decoder = Dense(flatten_dim)(decoder)
        else:
            decoder = Dense(flatten_dim)(latent_input)
        decoder = Reshape(new_shape)(decoder)
        for layer_dim in self.decoder_layers:
            decoder = Conv2D(layer_dim, kernel, activation='relu', padding='same')(decoder)
            decoder = UpSampling2D(pool)(decoder)
        decoder = Conv2D(1, kernel, activation='sigmoid', padding='same')(decoder)
        decoder = Model(latent_input, decoder) # define decoder

        # sampling
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
        eps = Input(tensor=K.random_normal(stddev=self.epsilon_std,
                                           shape=(K.shape(x)[0], self.latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])
        x_pred = decoder(z)
        autoencoder = Model([x, eps], x_pred) # add decoder sequentially
        encoder = Model(x, z_mu) # define encoder
        return autoencoder, encoder, decoder
