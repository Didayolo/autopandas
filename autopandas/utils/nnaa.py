import numpy as np
#from .metric import distance
'''
def distance():
    pass

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
