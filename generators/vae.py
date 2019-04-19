# Variational Autoencoder
# Inspired from: http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.models import Model, Sequential

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

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

class VAE():
    """ Variational Autoencoder
    """
    def __init__(self, original_dim, intermediate_dim=256, latent_dim=2, epsilon_std=1.0):
        decoder = Sequential([
            Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
            Dense(original_dim, activation='sigmoid')
        ])

        x = Input(shape=(original_dim,))
        h = Dense(intermediate_dim, activation='relu')(x)

        z_mu = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

        eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                           shape=(K.shape(x)[0], latent_dim)))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        x_pred = decoder(z)

        vae = Model(inputs=[x, eps], outputs=x_pred)
        encoder = Model(x, z_mu)
        vae.compile(optimizer='rmsprop', loss=nll)

        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

    def get_vae(self):
        return self.vae

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def fit(self, X, **kwargs):
        if isinstance(X, pd.DataFrame):
            X = X.as_matrix()
        return self.vae.fit(X, X, **kwargs)

    def sample(self, n=100, loc=0, scale=1):
        randoms = np.array([np.random.normal(loc, scale, self.latent_dim) for _ in range(n)])
        return self.decoder.predict(randoms)
