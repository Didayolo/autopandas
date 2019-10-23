# Autoencoder

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
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
    def __init__(self, input_dim, layers=[], latent_dim=2, architecture='fully', loss='nll', optimizer='rmsprop'):
        """ Autoencoder with fully connected layers.

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
        else:
            loss_function = _binary_crossentropy
        autoencoder.compile(optimizer=optimizer, loss=loss_function)
        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder

    def init_model(self, architecture='fully'):
        """ :param architecture: 'fully', 'cnn'
        """
        if architecture in ['fully', 'fc']:
            return self._init_model_fc()
        elif architecture in ['cnn', 'CNN']:
            return self._init_model_cnn()
        else:
            raise Exception('Unknown architecture: {}'.format(architecture))

    def _init_model_fc(self):
        """ Initialize fully connected architecture.
        """
        # encoder architecture
        input = Input(shape=(self.input_dim,))
        x = input
        if len(self.layers) > 0:
            x = Dense(self.layers[0], activation='relu')(x)
            for layer_dim in self.layers[1:]:
                x = Dense(layer_dim, activation='relu')(x)
        x = Dense(self.latent_dim, activation='relu')(x)
        z = x
        # decoder architecture
        if len(self.layers) > 0: # 1 or more intermediate layers
            # in the decoder we arrange layers in the opposite order compared to the encoder
            x = Dense(self.layers[-1], input_dim=self.latent_dim, activation='relu')(x)
            for layer_dim in reversed(self.layers[:-1]):
                x = Dense(layer_dim, activation='relu')(x)
        # else, no layers between input and latent space
        x = Dense(self.input_dim, activation='sigmoid')(x)
        autoencoder = Model(input, x) # define autoencoder
        encoder = Model(input, z) # define encoder
        # define decoder
        latent_input = Input(shape=(self.latent_dim,))
        decoder = latent_input
        for i in range((len(self.layers)+1)*-1, 0):
            decoder = autoencoder.layers[i](decoder)
        decoder = Model(latent_input, decoder)
        return autoencoder, encoder, decoder

    def _init_model_cnn(self):
        """ Initialize CNN architecture.
        """
        print('WARNING: CNN architecture is currently hard-coded for MNIST dataset.')
        kernel = (3, 3)
        pool = (2, 2)
        strides = (2, 2)

        # encoder architecture
        input = Input(shape=self.input_dim)
        x = input
        x = Conv2D(32, kernel, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool, padding='same')(x)
        x = Conv2D(8, kernel, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool, padding='same')(x)
        # flatten encoding for visualization
        x = Flatten()(x)
        z = x # latent space
        self.latent_dim = 7*7*8
        # decoder architecture
        x = Reshape((7, 7, 8))(x)
        x = Conv2D(8, kernel, activation='relu', padding='same')(x)
        x = UpSampling2D(pool)(x)
        x = Conv2D(32, kernel, activation='relu', padding='same')(x)
        x = UpSampling2D(pool)(x)
        x = Conv2D(1, kernel, activation='sigmoid', padding='same')(x)

        # define models
        autoencoder = Model(input, x)
        encoder = Model(input, z)
        latent_input = Input(shape=(self.latent_dim,))
        decoder = latent_input
        for i in range(-6, 0):
            decoder = autoencoder.layers[i](decoder)
        decoder = Model(latent_input, decoder)
        return autoencoder, encoder, decoder

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
        try:
            decoded = autopandas.AutoData(decoded)
            if self.columns is not None:
                decoded.columns = self.columns
            if self.indexes is not None:
                decoded.indexes = self.indexes
        except:
            print('WARNING: Impossible to cast sampled data to autopandas.AutoData')
        return decoded

    def siamese_distance(self, x, y, **kwargs):
        x_enc = self.encoder.predict(np.array([x]))
        y_enc = self.encoder.predict(np.array([y]))
        return autopandas.metric.distance(x_enc, y_enc, **kwargs)

    def distance(self, X, Y, **kwargs):
        """ Step 1: project X and Y in the learned latent space,
            Step 2: compute distance between the projections (NNAA score by default).
        """
        X_enc = self.encoder.predict(X)
        Y_enc = self.encoder.predict(Y)
        if not isinstance(X_enc, autopandas.AutoData):
            X_enc = autopandas.AutoData(X_enc)
        if not isinstance(Y_enc, autopandas.AutoData):
            Y_enc = autopandas.AutoData(Y_enc)
        return X_enc.distance(Y_enc, **kwargs)
