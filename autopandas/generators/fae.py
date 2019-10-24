# Fractal Autoencoder

from .autoencoder import AE
from .vae import VAE
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
import autopandas
import pandas as pd
import numpy as np

def merge(model1, model2):
    return model2(model1)

class FAE(AE):
    def __init__(self, layers, normalization=False, **kwargs):
        """ Fractal Autoencoder.
            AE with submodel training.

            :param layers: Dimension list of layers including input, intermediate (at least one) and latent layer.
        """
        self.latent_dim = layers[-1] # backward compatibility with VAE
        # create smaller models
        self.models = []
        for i in range(len(layers)-1):
            if i == len(layers) - 2: # last submodel
                submodel = VAE(layers[i], latent_dim=layers[i+1], **kwargs)
            else:
                submodel = AE(layers[i], latent_dim=layers[i+1], **kwargs)
            self.models.append(submodel)
        # TODO ########
        # merge encoders
        #encoder = self.models[0].encoder # normal loop
        #encoder = encoder()(Input(shape=encoder.layers[0].input_shape))
        #for i in range(len(self.models)-1):
            #encoder = merge(encoder, self.models[i+1].encoder)
        # merge decoders
        #decoder = self.models[-1].decoder # backward loop
        #decoder = decoder()(Input(shape=decoder.layers[0].input_shape))
        #for i in range(len(self.models)-2, -1, -1):
        #    decoder = merge(decoder, self.models[i-1].decoder)
        #self.encoder = encoder
        #self.decoder = decoder
        #self.autoencoder = None#merge(self.encoder, self.decoder) # complete model
        # for data frame indexes
        #autoencoder.compile(optimizer='rmsprop', loss=_nll)
        ###############
        # normalization
        self.normalization = normalization
        self.mins = None
        self.maxs = None
        # autopandas
        self.columns = None
        self.indexes = None

    def reset_normalization(self):
        if self.normalization:
            self.mins = []
            self.maxs = []
        else:
            self.mins = None
            self.maxs = None

    def normalize(self, X, i=None):
        # TODO autopandas normalize
        if self.normalization:
            if i is None: # fit
                self.mins.append(X.min())
                self.maxs.append(X.max())
                i = -1
            return (X - self.mins[i]) / (self.maxs[i] - self.mins[i]) # for nll loss
        else:
            return X

    def fit(self, X, epochs=10, validation_data=None, **kwargs):
        # epochs is an integer or a list with the same size as self.models
        # validation_data is a tuple (x_test, x_test)
        # fit each sub-model
        self.reset_normalization()
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
            if isinstance(X, autopandas.AutoData):
                self.indexes = X.indexes
            X = X.as_matrix()
        for i, model in enumerate(self.models):
            print('Training model {}/{}'.format(i+1, len(self.models)))
            print('Input shape: {}'.format(X.shape))
            if not isinstance(epochs, int): # different epochs number for each model
                ep = epochs[i]
            model.autoencoder.fit(X, X, epochs=ep, validation_data=validation_data, **kwargs) # train
            X = model.encoder.predict(X) # transform data for next submodel
            X = self.normalize(X)
            # update validation data
            if validation_data is not None:
                X_test = model.encoder.predict(validation_data[0])
                X_test = self.normalize(X_test)
                validation_data = (X_test, X_test)

    def encode(self, X):
        for i in range(len(self.models)):
            X = self.models[i].encoder.predict(X)
            X = self.normalize(X, i)
        return X

    def decode(self, X):
        for i in range(len(self.models)-1, -1, -1):
            X = self.models[i].decoder.predict(X)
            X = self.normalize(X, i)
        return X

    def autoencode(self, X):
        return self.decode(self.encode(X))

    def sample(self, n=100, loc=0, scale=1):
        """ :param scale: Standard deviation of gaussian distribution prior.
        """
        randoms = np.array([np.random.normal(loc, scale, self.latent_dim) for _ in range(n)])
        decoded = self.decode(randoms)
        decoded = autopandas.AutoData(decoded)
        if self.columns is not None:
            decoded.columns = self.columns
        if self.indexes is not None:
            decoded.indexes = self.indexes
        return decoded
