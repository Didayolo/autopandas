# Copula generator: TODO

from scipy.stats import gaussian_kde, norm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KernelDensity
import numpy as np

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
    ''' Retrofit the marginal distributions of the features in Xartif to those in Xreal.
    '''
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

def copula_generate(X):
    print('X marginals to uniforms...')
    X = matrix_to_rank(X)
    print('X uniforms to inverse gaussian cdf...')
    X = rank_matrix_to_inverse(X)
    print('Gaussian Kernel Density Estimation...')
    kernel = KernelDensity().fit(X)
    print('Generating artificial data \n Sampling from KDE distribution...')
    X_artif = kernel.sample(X.shape[0])
    print('Marginal retrofitting...')
    return marginal_retrofit(X_artif, X)
