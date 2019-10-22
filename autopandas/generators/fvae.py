# Fractal Variational Autoencoder

from .vae import VAE, nll
from keras import Input
from keras.models import Model, Sequential
import autopandas
import pandas as pd
import numpy as np

def merge(model1, model2):
    return model2(model1)

class FVAE(VAE):
    def __init__(self, layers):
        """ Variational Autoencoder.

            :param layers: Dimension list of layers including input, intermediate (at least one) and latent layer.
        """
        self.latent_dim = layers[-1] # backward compatibility with VAE
        # create smaller models
        self.models = []
        for i in range(len(layers)-1):
            submodel = VAE(layers[i], latent_dim=layers[i+1])
            self.models.append(submodel)
        # TODO ########
        # merge encoders
        #encoder = self.models[0].get_encoder() # normal loop
        #encoder = encoder()(Input(shape=encoder.layers[0].input_shape))
        #for i in range(len(self.models)-1):
            #encoder = merge(encoder, self.models[i+1].get_encoder())
        # merge decoders
        #decoder = self.models[-1].get_decoder() # backward loop
        #decoder = decoder()(Input(shape=decoder.layers[0].input_shape))
        #for i in range(len(self.models)-2, -1, -1):
        #    decoder = merge(decoder, self.models[i-1].get_decoder())
        #self.encoder = encoder
        #self.decoder = decoder
        #self.vae = None#merge(self.encoder, self.decoder) # complete model
        # for data frame indexes
        #vae.compile(optimizer='rmsprop', loss=nll)
        ###############
        self.columns = None
        self.indexes = None

    def get_vae(self):
        return self.vae

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def fit(self, X, epochs=10, validation_data=None, **kwargs):
        # epochs is an integer or a list with the same size as self.models
        # validation_data is a tuple (x_test, x_test)
        # fit each sub-model
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
            model.vae.fit(X, X, epochs=ep, validation_data=validation_data, **kwargs) # train
            X = model.get_encoder().predict(X) # transform data for next submodel
            X = (X + X.min()) / X.max() # for test purpose
            # update validation data
            if validation_data is not None:
                X_test = model.get_encoder().predict(validation_data[0])
                validation_data = (X_test, X_test)

    def encode(self, X):
        for model in self.models:
            X = model.get_encoder().predict(X)
        return X

    def decode(self, X):
        for i in range(len(self.models)-1, -1, -1):
            X = self.models[i].get_decoder().predict(X)
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
