# Distance metric functions

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import ks_2samp
from sklearn.utils import resample, shuffle
from sklearn.neighbors import NearestNeighbors
import itertools
from .nn_adversarial_accuracy import NearestNeighborMetrics

def distance(x, y, axis=None, norm='manhattan'):
    """
        Compute the distance between x and y.

        :param x: Array-like, first point
        :param y: Array-like, second point
        :param axis: Axis of x along which to compute the vector norms.
        :param norm: 'l0', 'manhattan', 'euclidean', 'minimum', 'maximum'
        :return: Distance value
        :rtype: float
    """
    z = x - y

    # if x and y are single values
    if not isinstance(x, (list, np.ndarray)):
        z = [x - y]

    if norm == 'manhattan' or distance == 'l1':
        return np.linalg.norm(z, ord=1, axis=axis)
    elif norm == 'euclidean' or distance == 'l2':
        return np.linalg.norm(z, ord=2, axis=axis)
    elif norm == 'minimum':
        return np.linalg.norm(z, ord=-np.inf, axis=axis)
    elif norm == 'maximum':
        return np.linalg.norm(z, ord=np.inf, axis=axis)
    elif norm == 'l0':
        return np.linalg.norm(z, ord=0, axis=axis)
    else:
        raise ValueError('Argument norm is invalid.')

def adversarial_accuracy(train, test, synthetics):
    """ Compute nearest neighbors adversarial accuracy metric
    """
    nnm = NearestNeighborMetrics(train, test, synthetics)
    nnm.compute_nn()
    adversarial = nnm.compute_adversarial_accuracy()
    return adversarial

def distance_correlation(X, Y):
    """
        Compute the distance correlation function.
        Works with X and Y of different dimensions (but same number of samples mandatory).

        :param X: Data
        :param y: Class data
        :return: Distance correlation
        :rtype: float
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def relief_divergence(X1, X2):
    """ Divergence based on ( dist_to_nearest_miss - dist_to_nearest_hit )"""
    p1, n = X1.shape
    p2, nn = X2.shape
    assert(n==nn)
    # Compute Euclidean distance between all pairs of examples, 1st matrix
    D1= squareform(pdist(X1))
    np.fill_diagonal(D1, float('Inf'))
    # Find distance to nearest hit
    nh = D1.min(1)
    # Compute Euclidean distance between all samples in X1 and X2
    D12=cdist(X1,X2)
    R = np.max(D12)
    nm = D12.min(1)
    # Mean difference dist to nearest miss and dist to nearest hit
    L = np.mean((nm - nh) / R)
    return max(0, L)

def acc_stat (solution, prediction):
    """ Return accuracy statistics TN, FP, TP, FN
     Assumes that solution and prediction are binary 0/1 vectors."""
     # This uses floats so the results are floats
    TN = sum(np.multiply((1-solution), (1-prediction)))
    FN = sum(np.multiply(solution, (1-prediction)))
    TP = sum(np.multiply(solution, prediction))
    FP = sum(np.multiply((1-solution), prediction))
    #print "TN =",TN
    #print "FP =",FP
    #print "TP =",TP
    #print "FN =",FN
    return (TN, FP, TP, FN)

def bac_metric (solution, prediction):
    """ Compute the balanced accuracy for binary classification. """
    [tn,fp,tp,fn] = acc_stat(solution, prediction)
    # Bounding to avoid division by 0
    eps = 1e-15
    tp = sp.maximum (eps, tp)
    pos_num = sp.maximum (eps, tp+fn)
    tpr = tp / pos_num # true positive rate (sensitivity)
    tn = sp.maximum (eps, tn)
    neg_num = sp.maximum (eps, tn+fp)
    tnr = tn / neg_num # true negative rate (specificity)
    bac = 0.5*(tpr + tnr)
    return bac

def nn_discrepancy(X1, X2):
    """Use 1 nearest neighbor method to determine discrepancy X1 and X2.
    If X1 and X2 are very different, it is easy to classify them
    thus bac > 0.5. Otherwise, if they are similar, bac ~ 0.5."""
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    X = np.concatenate((X1, X2))
    Y = np.concatenate((np.ones(n1), np.zeros(n2)))
    X, Y = shuffle(X, Y)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(X)
    distances, indices = nbrs.kneighbors(X)
    Ypred = Y[indices[:,1]] # the second nearest neighbor is the loo neighbor
    return max(0, 2*bac_metric(Y, Ypred)-1)

def ks_test(X1, X2):
    """ Paired Kolmogorov-Smirnov test for all matched pairs of variables in matrices X1 and X2."""
    n =X1.shape[1]
    ks=np.zeros(n)
    pval=np.zeros(n)
    for i in range(n):
        ks[i], pval[i] = ks_2samp (X1[:,i], X2[:,i])
    return (ks, pval)

def maximum_mean_discrepancy(A, B):
        """Compute the mean_discrepancy statistic between x and y"""
        # TODO...
        X = np.concatenate((A, B))
        #X = th.cat([x, y], 0)
        # dot product between all combinations of rows in 'X'
        #XX = X @ X.t()
        # dot product of rows with themselves
        # Old code : X2 = (X * X).sum(dim=1)
        X2 = (X * X).sum(dim=1)
        #X2 = XX.diag().unsqueeze(0)
        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        # -0.5 * (i^Ti - 2*i^Tj + j^Tj)

        #exponent = XX - 0.5 * (X2.expand_as(XX) + X2.t().expand_as(XX))

        #lossMMD = np.sum(self.S * sum([(exponent * (1./bandwith)).exp() for bandwith in self.bandwiths]))
        #L=lossMMD.sqrt()
        L = []
        return L

def cov_discrepancy(A, B):
    """Root mean square difference in covariance matrices"""
    CA = np.cov(A, rowvar=False);
    CB = np.cov(B, rowvar=False);
    n = CA.shape[0]
    L = np.sqrt( np.linalg.norm(CA-CB) / n**2 )
    return L

def corr_discrepancy(A, B):
    """Root mean square difference in correlation matrices"""
    CA = np.corrcoef(A, rowvar=False);
    CB = np.corrcoef(B, rowvar=False);
    n = CA.shape[0]
    L = np.sqrt( np.linalg.norm(CA-CB) / n**2 )
    return L
