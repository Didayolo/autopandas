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
# Still in utilities.py
# Find distances that works well with binary/categorical data

def distance(x, y, axis=None, norm='euclidean', method=None):
    """ Compute the distance between x and y (data points).

        :param x: Array-like, first point
        :param y: Array-like, second point
        :param axis: Axis of x along which to compute the vector norms.
        :param norm: 'l0', 'manhattan', 'euclidean', 'minimum' or 'maximum'
        :param method: Alias for norm parameter.
        :return: Distance value
        :rtype: float
    """
    if method is not None: # Alias
        norm = method
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

'''
# adversarial_accuracy
def nnaa(data1, data2, distance_func=None):
    """ Compute nearest neighbors adversarial accuracy
        Formula
        Can be seen as the binary classification score of a 1NN trying to tell if a point is from data1 or data2, in a leave one out setting.
    """
    data1, data2 = np.array(data1), np.array(data2)
    if distance_func is None:
        distance_func = distance
    len1, len2 = len(data1), len(data2)
    distance_matrix = np.empty((len1, len2))
    for i in range(len1):
        for j in range(len2):
            distance_matrix[i, j] = distance_func(data1[i], data2[j])
    HOT = distance_matrix.min(axis=0), distance_matrix.min(axis=1)
    # besoin de dist(d1, d1) et dist(d2, d2)
    return distance_matrix
'''
