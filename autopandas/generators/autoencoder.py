# Autoencoder

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
import autopandas

def _nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(binary_crossentropy(y_true, y_pred), axis=-1)

def _mse(y_true, y_pred):
    return mse(y_true, y_pred) # to pass loss to compile function

def _binary_crossentropy(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) # to pass loss to compile function

class AE():
    def __init__(self, input_dim, layers=[], latent_dim=2, loss='nll', optimizer='rmsprop'):
        """ Autoencoder with fully connected layers.

            :param input_dim: Input/output size.
            :param layers: Dimension of intermediate layers (encoder and decoder).
                                     It can be:
                                     - an integer (one intermediate layer)
                                     - a list of integers (several intermediate layers)
            :param latent_dim: Dimension of latent space layer.
            :param espilon_std: Standard deviation of gaussian distribution prior.
        """
        if isinstance(layers, int): # 1 intermediate layers
            layers = [layers]

        # encoder architecture
        input = Input(shape=(input_dim,))
        x = input
        if len(layers) > 0:
            x = Dense(layers[0], activation='relu')(x)
            for layer_dim in layers[1:]:
                x = Dense(layer_dim, activation='relu')(x)
        x = Dense(latent_dim, activation='relu')(x)
        z = x

        # decoder architecture
        if len(layers) > 0: # 1 or more intermediate layers
            # in the decoder we arrange layers in the opposite order compared to the encoder
            x = Dense(layers[-1], input_dim=latent_dim, activation='relu')(x)
            for layer_dim in reversed(layers[:-1]):
                x = Dense(layer_dim, activation='relu')(x)
        # else, no layers between input and latent space
        x = Dense(input_dim, activation='sigmoid')(x)

        # loss function
        if loss == 'nll':
            loss_function = _nll
        elif loss == 'mse':
            loss_function = _mse
        else:
            loss_function = _binary_crossentropy

        # define autoencoder
        self.autoencoder = Model(input, x)
        self.autoencoder.compile(optimizer=optimizer, loss=loss_function)

        # define encoder
        self.encoder = Model(input, z)
        # definer decoder
        latent_input = Input(shape=(latent_dim,))
        decoder = latent_input
        for i in range((len(layers)+1)*-1, 0):
            decoder = self.autoencoder.layers[i](decoder)
        self.decoder = Model(latent_input, decoder)
        # attributes
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # for data frame indexes
        self.columns = None
        self.indexes = None

    def get_autoencoder(self):
        return self.autoencoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def fit(self, X, **kwargs):
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
            if isinstance(X, autopandas.AutoData):
                self.indexes = X.indexes
            X = X.as_matrix()
        return self.autoencoder.fit(X, X, **kwargs)

    def sample(self, n=100, loc=0, scale=1):
        """ :param scale: Standard deviation of gaussian distribution prior.
        """
        randoms = np.array([np.random.normal(loc, scale, self.latent_dim) for _ in range(n)])
        decoded = self.decoder.predict(randoms)
        decoded = autopandas.AutoData(decoded)
        if self.columns is not None:
            decoded.columns = self.columns
        if self.indexes is not None:
            decoded.indexes = self.indexes
        return decoded
