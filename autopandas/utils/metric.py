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
#from .nnaa import nnaa

# Between Points/Columns (1D)
#############################
# A lot of functions are still in utilities.py (not imported module)
# TODO: Find distances that works well with binary/categorical data

def distance(x, y, axis=None, norm='euclidean', method=None):
    """ Compute the distance between x and y (data points).

        Default behaviour: flatten multi-dimensional arrays.

        :param x: Array-like, first point
        :param y: Array-like, second point
        :param axis: Axis of x along which to compute the vector norms.
        :param norm: 'l0', 'manhattan', 'euclidean', 'minimum' or 'maximum'
        :param method: Alias for norm parameter.
        :return: Distance value
        :rtype: float
    """
    if type(x) != type(y):
        raise Exception('x type is {} and y type is {}. Please pass two arguments with the same type.'.format(type(x), type(y)))
    if method is not None: # Alias
        norm = method
    # if x and y are single values
    if isinstance(x, np.ndarray):
        if len(x.shape) > 1:
            x = x.flatten()
        if len(y.shape) > 1:
            y = y.flatten()
        z = x - y
    elif isinstance(x, list):
        z = x - y
    else:
        z = [x - y]
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

def acc_stat(solution, prediction):
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

def bac_metric(solution, prediction):
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

# Between Distributions (2D)
############################

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

def discriminant(data1, data2, model=None, metric=None, name1='Dataset 1', name2='Dataset 2', same_size=False, verbose=False):
    """ Return the scores of a classifier trained to differentiate data1 and data2.

        If the distributions are similar and the model can't distinguish then the score will be ~ 0.5 (depending on the metric of course).

        :param model: The classifier. It has to have fit(X,y) and score(X,y) methods. Logistic regression by default.
        :param metric: The scoring metric. Accuracy by default.
        :param same_size: If True, normalize datasets to same size before computation.
        :return: Classification report (precision, recall, f1-score).
        :rtype: str
    """
    if model is None:
        model = LogisticRegression()
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
        # We want same number of example in both dataset to compute
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
    # If verbose, more information
    target_names = [name1, name2]
    model_info = str(model)
    report = classification_report(model.predict(X_test), y_test, target_names=target_names)
    if verbose:
        print(model_info)
        print(report)
        print('Metric: {}'.format(metric))
    # Scoring
    score = metric(y_test, model.predict(X_test))
    return score

def distance_matrix(data1, data2, distance_func=None):
    """ Compute matrix with distances between each points (m_ij is distance between i and j).
        TODO: parallelization.

        :param data1: Distribution
        :param data2: Distribution
        :param distance_func: Distance metric function to use to compare data points. Euclidean distance by default.
    """
    # distance metric between data points
    if distance_func is None:
        distance_func = distance
    len1, len2 = len(data1), len(data2) # handle len1 != len2 case?

    distance_matrix = np.empty((len1, len2))
    # compute the distances
    for i in range(len1):
        for j in range(len2):
            distance_matrix[i, j] = distance_func(data1[i], data2[j])
    return distance_matrix

def nnaa(data_s, data_t, distance_func=None, detailed_results=False):
    """ Compute nearest neighbors adversarial accuracy between data_s and data_t.
        This is the proportion of points in data_s for which the nearest neighbor is in data_s (and not in data_t).
        It can also be seen as the binary classification score of a 1NN trying to tell if a point is from data1 or data2, in a leave one out setting.
        If data_s and data_t follow the same distribution, the score should be near 0.5:
        * nnaa > 0.5 -> underfitting
        * nnaa ~ 0.5 -> cool
        * nnaa < 0.5 -> overfitting

        From "Privacy Preserving Synthetic Health Data" by Andrew Yale et al.

        :param data_s: 2D distribution (s for "source").
        :param data_t: 2D distribution (t for "target").
        :param distance_func: Distance metric function to use to compare data points. Euclidean distance by default.
        :param detailed_results: If True, return score but also score for TS and ST (the 2 components of the score).
    """
    data_s, data_t = np.array(data_s), np.array(data_t)
    if(data_s.shape != data_t.shape):
        raise Exception('The two data frames must have the same shapes but {} != {} got passed.'.format(data_s.shape, data_t.shape))
    # matrixes with distances between each points (m_ij is distance between i and j)
    distances_st = distance_matrix(data_s, data_t, distance_func=distance_func) # distances between data_s and data_t
    distances_ss = distance_matrix(data_s, data_s, distance_func=distance_func)
    distances_tt = distance_matrix(data_t, data_t, distance_func=distance_func)
    # fill the diagonal to avoid considering the points themselves as their nearest neighbors
    np.fill_diagonal(distances_ss, np.inf)
    np.fill_diagonal(distances_tt, np.inf)
    # distance to nearest neighbors
    min_st, min_ts = distances_st.min(axis=0), distances_st.min(axis=1)
    min_ss = distances_ss.min(axis=0)
    min_tt = distances_tt.min(axis=0)
    nnaa_st = np.sum(min_st > min_ss) / len(data_s) # proportion of nearest neihbors of s in s
    nnaa_ts = np.sum(min_ts > min_tt) / len(data_t)
    score = (nnaa_st + nnaa_ts) / 2
    if detailed_results:
        return score, nnaa_st, nnaa_ts
    else:
        return score
