# Distance metric functions

import numpy as np
import scipy as sp
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import ks_2samp
from sklearn.utils import resample, shuffle
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import itertools
#from .nn_adversarial_accuracy import NearestNeighborMetrics
from .nnaa import nnaa

def distance(x, y, axis=None, norm='euclidean'):
    """ Compute the distance between x and y.
        :param x: Array-like, first point
        :param y: Array-like, second point
        :param axis: Axis of x along which to compute the vector norms.
        :param norm: 'l0', 'manhattan', 'euclidean', 'minimum' or 'maximum'
        :return: Distance value
        :rtype: float
    """
    # if x and y are single values
    if not isinstance(x, (list, np.ndarray)):
        z = [x - y]
    else:
        z = x - y
    if norm == 'manhattan' or norm == 'l1':
        return np.linalg.norm(z, ord=1, axis=axis)
    elif norm == 'euclidean' or norm == 'l2':
        return np.linalg.norm(z, ord=2, axis=axis)
    elif norm == 'minimum':
        return np.linalg.norm(z, ord=-np.inf, axis=axis)
    elif norm == 'maximum':
        return np.linalg.norm(z, ord=np.inf, axis=axis)
    elif norm == 'l0':
        return np.linalg.norm(z, ord=0, axis=axis)
    else:
        raise ValueError('Unknwon norm: {}.'.format(norm))

#def adversarial_accuracy(train, test, synthetics):
#    """ Compute nearest neighbors adversarial accuracy metric
#    """
#    nnm = NearestNeighborMetrics(train, test, synthetics)
#    nnm.compute_nn()
#    adversarial = nnm.compute_adversarial_accuracy()
#    return adversarial

#def tmp_aa(data1, data2):
#    """ New implementation of AA metric
#    """
#    # compute all distances
#    # ...
#    # 2 times quicker than naive 1NN leave-one-out
#    pass

def distance_correlation(X, Y):
    """ Compute the distance correlation function.
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
    """ Divergence based on ( dist_to_nearest_miss - dist_to_nearest_hit )
    """
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
        Assumes that solution and prediction are binary 0/1 vectors.
     """
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
    """ Compute the balanced accuracy for binary classification.
    """
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
    """ Use 1 nearest neighbor method to determine discrepancy between X1 and X2.
        If X1 and X2 are very different, it is easy to classify them
        thus bac > 0.5. Otherwise, if they are similar, bac ~ 0.5.
    """
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
    """ Paired Kolmogorov-Smirnov test for all matched pairs of variables in matrices X1 and X2.
    """
    n =X1.shape[1]
    ks=np.zeros(n)
    pval=np.zeros(n)
    for i in range(n):
        ks[i], pval[i] = ks_2samp (X1[:,i], X2[:,i])
    return (ks, pval)

def maximum_mean_discrepancy(A, B):
        """ Compute the mean_discrepancy statistic between x and y.
        """
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
    """ Root mean square difference in covariance matrices.
    """
    CA = np.cov(A, rowvar=False);
    CB = np.cov(B, rowvar=False);
    n = CA.shape[0]
    L = np.sqrt( np.linalg.norm(CA-CB) / n**2 )
    return L

def corr_discrepancy(A, B):
    """ Root mean square difference in correlation matrices.
    """
    CA = np.corrcoef(A, rowvar=False);
    CB = np.corrcoef(B, rowvar=False);
    n = CA.shape[0]
    L = np.sqrt( np.linalg.norm(CA-CB) / n**2 )
    return L

def discriminant(data1, data2, model=LogisticRegression(), metric=None, name1='Dataset 1', name2='Dataset 2', same_size=False, verbose=False):
    """ Return the scores of a classifier trained to differentiate data1 and data2.

        :param model: The classifier. It has to have fit(X,y) and score(X,y) methods.
        :param metric: The scoring metric
        :param same_size: If True, normalize datasets to same size before computation.
        :return: Classification report (precision, recall, f1-score).
        :rtype: str
    """
    if metric is None:
        metric = accuracy_score
    # check if train/test split already exists or do it
    if not data1.has_split():
        data1.train_test_split()
    if not data2.has_split():
        data2.train_test_split()
    ds1_train = data1.get_data('train')
    ds1_test = data1.get_data('test')
    ds2_train = data2.get_data('train')
    ds2_test = data2.get_data('test')
    if same_size:
        # Same number of example in both dataset to compute
        if ds1_train.shape[0] < ds2_train.shape[0]:
            ds2_train = ds2_train.sample(n=ds1_train.shape[0])
        if ds1_train.shape[0] > ds2_train.shape[0]:
            ds1_train = ds1_train.sample(n=ds1_train.shape[0])
    # Train set
    X1_train, X2_train = list(ds1_train.values), list(ds2_train.values)
    X_train = X1_train + X2_train
    y_train = [0] * len(X1_train) + [1] * len(X2_train)
    # Shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)
    # Test set
    X1_test, X2_test = list(ds1_test.values), list(ds2_test.values)
    X_test = X1_test + X2_test
    y_test = [0] * len(X1_test) + [1] * len(X2_test)
    # Training
    model.fit(X_train, y_train)
    # Score
    #clf.score(X_test, y_test)
    # metric here
    target_names = [name1, name2]
    model_info = str(model)
    report = classification_report(model.predict(X_test), y_test, target_names=target_names)
    if verbose:
        print(model_info)
        print(report)
        print('Metric: {}'.format(metric))
    return metric(y_test, model.predict(X_test))
