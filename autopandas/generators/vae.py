# Variational Autoencoder
# Inspired from: http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/

# Imports
from .autoencoder import AE, _nll, _mse, _binary_crossentropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
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
    def __init__(self, input_dim, layers=[], latent_dim=2, architecture='fully', epsilon_std=1.0, loss='nll', optimizer='rmsprop'):
        """ Variational Autoencoder.

            :param input_dim: Input/output size.
            :param layers: Dimension of intermediate layers (encoder and decoder).
                                     It can be:
                                     - an integer (one intermediate layer)
                                     - a list of integers (several intermediate layers)
            :param latent_dim: Dimension of latent space layer.
            :param architecture: 'fully', 'cnn'.
            :param espilon_std: Standard deviation of gaussian distribution prior.
        """
        if isinstance(layers, int): # 1 intermediate layers
            layers = [layers]
        # attributes
        self.input_dim = input_dim
        self.layers = layers
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        # for data frame indexes
        self.columns = None
        self.indexes = None
        # init architecture
        autoencoder, encoder, decoder = self.init_model(architecture=architecture)
        # loss function
        if loss == 'nll':
            loss_function = _nll
        elif loss == 'mse':
            loss_function = _mse
        elif loss == 'binary_crossentropy':
            loss_function = _binary_crossentropy
        else:
            raise Exception('Unknown loss name: {}'.format(loss))
        autoencoder.compile(optimizer=optimizer, loss=loss_function)
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder

    def _init_model_fc(self):
        """ Initialize fully connected architecture.
        """
        # encoder architecture
        x = Input(shape=(self.input_dim,))
        if len(self.layers) > 0:
            h = Dense(self.layers[0], activation='relu')(x)
            for layer_dim in self.layers[1:]:
                h = Dense(layer_dim, activation='relu')(h)
        else:
            h = x

        # decoder architecture
        decoder = Sequential()
        if len(self.layers) > 0: # 1 or more intermediate layers
            # in the decoder we arrange layers in the opposite order compared to the encoder
            decoder.add(Dense(self.layers[-1], input_dim=self.latent_dim, activation='relu'))
            for layer_dim in reversed(self.layers[:-1]):
                decoder.add(Dense(layer_dim, activation='relu'))
        # else, no layers between input and latent space
        decoder.add(Dense(self.input_dim, activation='sigmoid'))

        # sampling
        z_mu = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
        eps = Input(tensor=K.random_normal(stddev=self.epsilon_std,
                                           shape=(K.shape(x)[0], self.latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])
        x_pred = decoder(z)
        autoencoder = Model([x, eps], x_pred)
        encoder = Model(x, z_mu)
        return autoencoder, encoder, decoder

    def _init_model_cnn(self):
        """ Initialize CNN architecture.
        """
        print('WARNING: CNN architecture is currently hard-coded for MNIST dataset.')
        kernel = (3, 3)
        pool = (2, 2)
        strides = (2, 2)

        # encoder architecture
        x = Input(shape=self.input_dim)
        h = Conv2D(32, kernel, activation='relu', padding='same')(x)
        h = MaxPooling2D(pool, padding='same')(h)
        h = Conv2D(8, kernel, activation='relu', padding='same')(h)
        h = MaxPooling2D(pool, padding='same')(h)

        # flatten encoding for visualization (TODO)
        h = Flatten()(h)
        self.latent_dim = 7*7*8

        # decoder architecture
        latent_input = Input(shape=(self.latent_dim,))
        decoder = latent_input
        decoder = Reshape((7, 7, 8))(decoder)
        decoder = Conv2D(8, kernel, activation='relu', padding='same')(decoder)
        decoder = UpSampling2D(pool)(decoder)
        decoder = Conv2D(32, kernel, activation='relu', padding='same')(decoder)
        decoder = UpSampling2D(pool)(decoder)
        decoder = Conv2D(1, kernel, activation='sigmoid', padding='same')(decoder)

        # sampling
        z_mu = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)
        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)
        eps = Input(tensor=K.random_normal(stddev=self.epsilon_std,
                                           shape=(K.shape(x)[0], self.latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])
        decoder = Model(latent_input, decoder)
        x_pred = decoder(z)
        autoencoder = Model([x, eps], x_pred) # add decoder sequentially
        encoder = Model(x, z_mu)
        return autoencoder, encoder, decoder
