# Copula generator

from scipy.stats import gaussian_kde, norm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KernelDensity
import numpy as np
import autopandas

def vector_to_rank(x):
    sort = np.sort(x)
    rank = LabelEncoder().fit_transform(sort) + 1
    rank_dict = dict(zip(sort, rank))
    return [rank_dict[i] for i in x]

def matrix_to_rank(X):
    matrix = np.copy(X)
    for i, column in enumerate(matrix.T):
        matrix[:, i] = vector_to_rank(column)
    return matrix

def rank_vector_to_inverse(x):
    x = x / (x.max() + 1)
    inverse = norm.ppf(x)
    return inverse

def rank_matrix_to_inverse(X):
    matrix = np.copy(X)
    for i, column in enumerate(X.T):
        matrix[:, i] = rank_vector_to_inverse(column)
    return matrix

def marginal_retrofit(Xartif, Xreal):
    """ Retrofit the marginal distributions of the features in Xartif to those in Xreal.
    """
    pa,n = Xartif.shape
    pr,nr = Xreal.shape
    assert(n==nr)
    Xretro = np.zeros(Xartif.shape)
    # Adjust the dimensions of the 2 matrices
    if pa>pr:
        # subsample Xreal
        Xreal = resample(Xreal, replace='False', n_samples=pa)
    elif pa<pr:
        # oversample Xreal
        Xreal = resample(Xreal, replace='True', n_samples=pa)
    # Otherwise do nothing
    # Loop over variables
    for i in range(n):
        # Sort the values of both arrays
        Xa=Xartif[:,i]
        Xr=Xreal[:,i]
        idxa=np.argsort(Xa)
        idxr=np.argsort(Xr)
        # Substitute artificial value for corresponding real value at same rank
        Xa[idxa] = Xr[idxr]
        # replace initial column
        Xretro[:,i] = Xa
    return Xretro

def copula_generate(X, generator=None, n=None):
    """ Generate using copula trick.

        :param generator: Model to fit and sample from. KDE by default.
        :param n: Number of examples to generate. By default it is the number of observations in X.
    """
    indexes = X.indexes
    columns = X.columns
    if generator is None:
        generator = KernelDensity()
    if n is None:
        n = X.shape[0]
    X_real = np.array(X)
    # X marginals to uniforms
    X = matrix_to_rank(X)
    # X uniforms to inverse gaussian CDF
    X = rank_matrix_to_inverse(X)
    # Fit generator
    generator.fit(X)
    # Generating artificial data \n Sampling from generator
    X_artif = generator.sample(n)
    # Marginal retrofitting
    result = autopandas.AutoData(marginal_retrofit(X_artif, X_real))
    # Restore data frame index
    result.indexes = indexes
    result.columns = columns
    return result

class Copula():
    def __init__(self):
        """ Copula generator.
        """
        self.data = None

    def fit(self, data):
        """ Use the copula trick and train the generator with data.

            :param data: Data frame to use as training set.
        """
        self.data = copula_generate(data)

    def sample(self, n=1, replace=False):
        """ Sample from trained generator.

            :param n: Number of examples to sample.
            :param replace: If True, sample with replacement.
        """
        if self.data is None:
            raise Exception('You firstly need to train the copula generator before sampling. Please use fit method.')
        else:
            return self.data.sample(n, replace=replace)
