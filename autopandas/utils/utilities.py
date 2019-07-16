# This file is not part of the project.
# We use it as a draft to copy and paste function from previous work.

# Useful functions for computation and visualization of data descriptors

# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from .metric import *
from copy import deepcopy # for preprocessing

# Hierarchical clustering
import matplotlib as mpl
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import string
import time
import sys, os
import getopt

# PCA, LDA, T-SNE
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

# Kolmogorov-Smirnov, Chi-square
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency

# KL divergence, mutual information, Jensen-Shannon
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from numpy.linalg import norm

# Area under curve
from scipy.integrate import simps
from numpy import trapz

# MMD
import torch as th
from torch.autograd import Variable

def printmd(string):
    """ Print Markdown string

        :param string: String to display.
    """
    display(Markdown(string))

def normalize(l, normalization='probability'):
    """ Return a normalized list
        Input:
          normalization: 'probability': between 0 and 1 with a sum equals to 1
                         'min-max': min become 0 and max become 1
    """
    if normalization=='probability':
        return [float(i)/sum(l) for i in l]
    elif normalization=='min-max':
        return [(float(i) - min(l)) / (max(l) - min(l)) for i in l]
    else: # mean std ?
        raise ValueError('Argument normalization is invalid.')

def show_classes(y):
    """ Shows classes distribution

		:param y: Pandas DataFrame representing classes
	"""
    for column in y.columns:
        sns.distplot(y[column])
        plt.show()

def show_correlation(df, size=10):
    """
        Shows a graphical correlation matrix for each pair of columns in the dataframe.
        :param df: Pandas DataFrame
        :param size: Vertical and horizontal size of the plot
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

def heatmap(X, row_method, column_method, row_metric, column_metric,
            color_gradient):

    print(
        "\nPerforming hierarchical clustering using {} for columns and {} for rows".
        format(column_metric, row_metric))
    """
    This below code is based in large part on the protype methods:
    http://old.nabble.com/How-to-plot-heatmap-with-matplotlib--td32534593.html
    http://stackoverflow.com/questions/7664826/how-to-get-flat-clustering-corresponding-to-color-clusters-in-the-dendrogram-cre

    x is an m by n ndarray, m observations, n genes
    """

    ### Define variables
    x = np.array(X)
    column_header = column_header = ['T' + str(dataset) for dataset in list(X)]  # X.columns.values
    row_header = ['A' + str(model) for model in list(X.index)]  # X.index

    ### Define the color gradient to use based on the provided name
    n = len(x[0])
    m = len(x)
    if color_gradient == 'red_white_blue':
        cmap = plt.cm.bwr
    if color_gradient == 'red_black_sky':
        cmap = RedBlackSkyBlue()
    if color_gradient == 'red_black_blue':
        cmap = RedBlackBlue()
    if color_gradient == 'red_black_green':
        cmap = RedBlackGreen()
    if color_gradient == 'yellow_black_blue':
        cmap = YellowBlackBlue()
    if color_gradient == 'seismic':
        cmap = plt.cm.seismic
    if color_gradient == 'green_white_purple':
        cmap = plt.cm.PiYG_r
    if color_gradient == 'coolwarm':
        cmap = plt.cm.coolwarm

    ### Scale the max and min colors so that 0 is white/black
    vmin = x.min()
    vmax = x.max()
    vmax = max([vmax, abs(vmin)])
    # vmin = vmax*-1
    # norm = mpl.colors.Normalize(vmin/2, vmax/2) ### adjust the max and min to scale these colors
    norm = mpl.colors.Normalize(vmin, vmax)
    ### Scale the Matplotlib window size
    default_window_hight = 8.5
    default_window_width = 12
    fig = plt.figure(figsize=(default_window_width, default_window_hight))
    ### could use m,n to scale here
    color_bar_w = 0.015  ### Sufficient size to show

    ## calculate positions for all elements
    # axm, placement of heatmap for the data matrix
    [axm_x, axm_y, axm_w, axm_h] = [0.05, 0.95, 1, 1]
    width_between_axm_axr = 0.01
    text_margin = 0.1 # space between color bar and feature names

    # axr, placement of row side colorbar
    [axr_x, axr_y, axr_w, axr_h] = [0.31, 0.1, color_bar_w, 0.6]
    ### second to last controls the width of the side color bar - 0.015 when showing
    axr_x =  axm_x + axm_w + width_between_axm_axr + text_margin
    axr_y = axm_y
    axr_h = axm_h
    width_between_axr_ax1 = 0.004

    # ax1, placement of dendrogram 1, on the right of the heatmap
    #if row_method != None: w1 =
    [ax1_x, ax1_y, ax1_w, ax1_h] = [0.05, 0.22, 0.2, 0.6]
    ax1_x = axr_x + axr_w + width_between_axr_ax1
    ax1_y = axr_y
    ax1_h = axr_h
    ### The second value controls the position of the matrix relative to the bottom of the view
    width_between_ax1_axr = 0.004
    height_between_ax1_axc = 0.004  ### distance between the top color bar axis and the matrix

    # axc, placement of column side colorbar
    [axc_x, axc_y, axc_w, axc_h] = [0.4, 0.63, 0.5, color_bar_w]
    ### last one controls the height of the top color bar - 0.015 when showing
    axc_x = axm_x
    axc_y = axm_y - axc_h - width_between_axm_axr - text_margin
    axc_w = axm_w
    height_between_axc_ax2 = 0.004

    # ax2, placement of dendrogram 2, on the top of the heatmap
    [ax2_x, ax2_y, ax2_w, ax2_h] = [0.3, 0.72, 0.6, 0.15]
    ### last one controls height of the dendrogram
    ax2_x = axc_x
    ax2_y = axc_y - axc_h - ax2_h - height_between_axc_ax2
    ax2_w = axc_w

    # axcb - placement of the color legend
    [axcb_x, axcb_y, axcb_w, axcb_h] = [0.07, 0.88, 0.18, 0.09]
    axcb_x = ax1_x
    axcb_y = ax2_y
    axcb_w = ax1_w
    axcb_h = ax2_h

    # Compute and plot bottom dendrogram
    if column_method != None:
        start_time = time.time()
        d2 = dist.pdist(x.T)
        D2 = dist.squareform(d2)
        ax2 = fig.add_axes([ax2_x, ax2_y, ax2_w, ax2_h], frame_on=True)
        Y2 = sch.linkage(D2, method=column_method, metric=column_metric)
        ### array-clustering metric - 'average', 'single', 'centroid', 'complete'
        Z2 = sch.dendrogram(Y2, orientation='bottom')
        ind2 = sch.fcluster(Y2, 0.7 * max(Y2[:, 2]), 'distance')
        ### This is the default behavior of dendrogram
        ax2.set_xticks([])  ### Hides ticks
        ax2.set_yticks([])
        time_diff = str(round(time.time() - start_time, 1))
        print('Column clustering completed in {} seconds'.format(time_diff))
    else:
        ind2 = ['NA'] * len(column_header)
        ### Used for exporting the flat cluster data

    # Compute and plot right dendrogram.
    if row_method != None:
        start_time = time.time()
        d1 = dist.pdist(x)
        D1 = dist.squareform(d1)  # full matrix
        ax1 = fig.add_axes([ax1_x, ax1_y, ax1_w, ax1_h], frame_on=True)
        # frame_on may be False
        Y1 = sch.linkage(D1, method=row_method, metric=row_metric)
        ### gene-clustering metric - 'average', 'single', 'centroid', 'complete'
        Z1 = sch.dendrogram(Y1, orientation='right')
        ind1 = sch.fcluster(Y1, 0.7 * max(Y1[:, 2]), 'distance')
        ### This is the default behavior of dendrogram
        # print 'ind1', ind1
        ax1.set_xticks([])  ### Hides ticks
        ax1.set_yticks([])
        time_diff = str(round(time.time() - start_time, 1))
        print('Row clustering completed in {} seconds'.format(time_diff))
    else:
        ind1 = ['NA'] * len(row_header)
        ### Used for exporting the flat cluster data

    # Plot distance matrix.
    axm = fig.add_axes([axm_x, axm_y, axm_w, axm_h])
    # axes for the data matrix
    xt = x
    if column_method != None:
        idx2 = Z2['leaves']
        ### apply the clustering for the array-dendrograms to the actual matrix data
        xt = xt[:, idx2]
        # print 'idx2', idx2, len(idx2)
        # print 'ind2', ind2, len(ind2)
        ind2 = [ind2[i] for i in idx2]
        # ind2 = ind2[:,idx2] ### reorder the flat cluster to match the order of the leaves the dendrogram
    if row_method != None:
        idx1 = Z1['leaves']
        ### apply the clustering for the gene-dendrograms to the actual matrix data
        xt = xt[idx1, :]  # xt is transformed x
        # ind1 = ind1[idx1,:] ### reorder the flat cluster to match the order of the leaves the dendrogram
        ind1 = [ind1[i] for i in idx1]
    ### taken from http://stackoverflow.com/questions/2982929/plotting-results-of-hierarchical-clustering-ontop-of-a-matrix-of-data-in-python/3011894#3011894
    # print xt
    im = axm.matshow(xt, aspect='auto', origin='lower', cmap=cmap, norm=norm)
    ### norm=norm added to scale coloring of expression with zero = white or black
    axm.set_xticks([])  ### Hides x-ticks
    axm.set_yticks([])

    # Add text
    new_row_header = []
    new_column_header = []
    for i in range(x.shape[0]):
        if row_method != None:
            if len(
                    row_header
            ) < 100:  ### Don't visualize gene associations when more than 100 rows
                axm.text(x.shape[1] - 0.5, i, '  ' + row_header[idx1[i]])
            new_row_header.append(row_header[idx1[i]])
        else:
            if len(
                    row_header
            ) < 100:  ### Don't visualize gene associations when more than 100 rows
                axm.text(x.shape[1] - 0.5, i,
                         '  ' + row_header[i])  ### When not clustering rows
            new_row_header.append(row_header[i])
    for i in range(x.shape[1]):
        if column_method != None:
            axm.text(
                i,
                -0.9,
                ' ' + column_header[idx2[i]],
                rotation=270,
                verticalalignment="top")  # rotation could also be degrees
            new_column_header.append(column_header[idx2[i]])
        else:  ### When not clustering columns
            axm.text(
                i,
                -0.9,
                ' ' + column_header[i],
                rotation=270,
                verticalalignment="top")
            new_column_header.append(column_header[i])

    for j in range(x.shape[0]):
        if row_method != None:
            axm.text(
                len(new_column_header) + 1,
                j,
                ' ' + row_header[idx1[j]],
                rotation=0,
                verticalalignment="top")  # rotation could also be degrees
            new_row_header.append(row_header[idx1[j]])
        else:  ### When not clustering columns
            axm.text(
                len(new_column_header) + 1,
                j,
                ' ' + row_header[j],
                rotation=0,
                verticalalignment="top")
            new_row_header.append(row_header[j])

    # Plot colside colors
    # axc --> axes for column side colorbar
    if column_method != None:
        axc = fig.add_axes([axc_x, axc_y, axc_w,
                            axc_h])  # axes for column side colorbar
        cmap_c = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
        dc = np.array(ind2, dtype=int)
        dc.shape = (1, len(ind2))
        im_c = axc.matshow(dc, aspect='auto', origin='lower', cmap=cmap_c)
        axc.set_xticks([])  ### Hides ticks
        axc.set_yticks([])

    # Plot rowside colors
    # axr --> axes for row side colorbar
    if row_method != None:
        axr = fig.add_axes([axr_x, axr_y, axr_w,
                            axr_h])  # axes for column side colorbar
        dr = np.array(ind1, dtype=int)
        dr.shape = (len(ind1), 1)
        #print ind1, len(ind1)
        cmap_r = mpl.colors.ListedColormap(['r', 'g', 'b', 'y', 'w', 'k', 'm'])
        im_r = axr.matshow(dr, aspect='auto', origin='lower', cmap=cmap_r)
        axr.set_xticks([])  ### Hides ticks
        axr.set_yticks([])

    # Plot color legend
    axcb = fig.add_axes(
        [axcb_x, axcb_y, axcb_w, axcb_h], frame_on=False)  # axes for colorbar
    # print 'axcb', axcb
    cb = mpl.colorbar.ColorbarBase(
        axcb, cmap=cmap, norm=norm, orientation='horizontal')
    # print cb
    axcb.set_title("colorkey")

    cb.set_label("Differential Expression (log2 fold)")

    ### Render the graphic
    if len(row_header) > 50 or len(column_header) > 50:
        plt.rcParams['font.size'] = 5
    else:
        plt.rcParams['font.size'] = 8

    plt.show()


def compute_pca(X, verbose=False, **kwargs):
    """
        Compute PCA.

        :param X: Data
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for PCA (see sklearn doc)
        :return: Tuple (pca, X) containing a PCA object (see sklearn doc) and the transformed data
        :rtype: Tuple
    """
    pca = PCA(**kwargs)
    X = pca.fit_transform(X)

    print('Explained variance ratio of the {} components: \n {}'.format(pca.n_components_,
                                                                        pca.explained_variance_ratio_))
    if verbose:
        plt.bar(left=range(pca.n_components_),
                height=pca.explained_variance_ratio_,
                width=0.3,
                tick_label=range(pca.n_components_))
        plt.title('Explained variance ratio by principal component')
        plt.show()

    return pca, X


def show_pca(X, y=None, i=1, j=2, verbose=False, **kwargs):
    """
        Plot PCA.

        :param X: Data
        :param y: Labels
        :param i: i_th component of the PCA
        :param j: j_th component of the PCA
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for PCA (see sklearn doc)
    """
    pca, X = compute_pca(X, verbose, **kwargs)

    assert(i <= pca.n_components_ and j <= pca.n_components_ and i != j)

    if y is not None:

        if isinstance (y, pd.DataFrame):
            target_names = y.columns.values
            y = y.values
        elif isinstance(y, pd.Series):
            target_names = y.unique()
            y = y.values
        else:
            target_names = np.unique(y)

        if len(y.shape) > 1:
            if y.shape[1] > 1:
                y = np.where(y==1)[1]

        for label in range(len(target_names)):
            plt.scatter(X[y == label, i-1], X[y == label, j-1], alpha=.8, lw=2, label=target_names[label])

        plt.legend(loc='best', shadow=False, scatterpoints=1)

    else:
        plt.scatter(X.T[0], X.T[1], alpha=.8, lw=2)

    plt.xlabel('PC '+str(i))
    plt.ylabel('PC '+str(j))
    plt.title('Principal Component Analysis: PC{} and PC{}'.format(str(i), str(j)))
    plt.show()


def compute_lda(X, y, verbose=False, **kwargs):
    """
        Compute LDA.

        :param X: Data
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for LDA (see sklearn doc)
        :return: Tuple (lda, X) containing a LDA object (see sklearn doc) and the transformed data
        :rtype: Tuple
    """
    lda = LinearDiscriminantAnalysis(**kwargs)
    X = lda.fit_transform(X, y)

    return lda, X


def show_lda(X, y, verbose=False, **kwargs):
    """
        Plot LDA.

        :param X: Data
        :param y: Labels
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for PCA (see sklearn doc)
    """
    if isinstance (y, pd.DataFrame):
        target_names = y.columns.values
        y = y.values
    elif isinstance(y, pd.Series):
        target_names = y.unique()
        y = y.values
    else:
        target_names = np.unique(y)

    # Flatten one-hot
    if len(y.shape) > 1:
        if y.shape[1] > 1:
            y = np.where(y==1)[1]

    _, X = compute_lda(X, y, verbose=verbose, **kwargs)

    for label in range(len(target_names)):
        plt.scatter(X[y == label, 0], X[y == label, 1], alpha=.8, lw=2, label=target_names[label])

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of dataset')
    plt.show()


def compute_tsne(X, verbose=False, **kwargs):
    """
        Compute T-SNE.

        :param X: Data
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for T-SNE (see sklearn doc)
        :return: Tuple (tsne, X) containing a T-SNE object (see sklearn doc) and the transformed data
        :rtype: Tuple
    """
    tsne = TSNE(**kwargs)
    X = tsne.fit_transform(X)

    return tsne, X


def show_tsne(X, y, i=1, j=2, verbose=False, **kwargs):
    """
        Plot T-SNE.

        :param X: Data
        :param y: Labels
        :param i: i_th component of the T-SNE
        :param j: j_th component of the T-SNE
        :param verbose: Display additional information during run
        :param **kwargs: Additional parameters for T-SNE (see sklearn doc)
    """
    tsne, X = compute_tsne(X, verbose=verbose, **kwargs)
    assert(i <= tsne.embedding_.shape[1] and j <= tsne.embedding_.shape[1] and i != j)

    if isinstance (y, pd.DataFrame):
        target_names = y.columns.values
        y = y.values
    elif isinstance(y, pd.Series):
        target_names = y.unique()
        y = y.values
    else:
        target_names = np.unique(y)

    if y.shape[1] > 1:
        y = np.where(y==1)[1]

    for label in range(len(target_names)):
        plt.scatter(X[y == label, i-1], X[y == label, j-1], alpha=.8, lw=2, label=target_names[label])
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('T-SNE: TC{} and TC{}'.format(str(i), str(j)))
    plt.show()


def minimum_distance(A, B, norm='manhattan'):
    """ Compute for each element of A its distance from its nearest neighbor from B (and reciprocally)

        :param A: Distribution A
        :param B: Distribution B
        :param norm: Norm used for distance computations

        :return: mdA: Distances of A samples nearest neighbors from B
        :return: mdB: Distances of B samples nearest neighbors from A
    """
    # Minimum distances
    mdA = [None for _ in range(len(A))]
    mdB = [None for _ in range(len(B))]

    for i in range(len(A)):
        for j in range(len(B)):

            d = distance(A[i], B[j], norm=norm)

            if mdA[i] == None or mdA[i] > d:
                mdA[i] = d

            if mdB[j] == None or mdB[j] > d:
                mdB[j] = d

    return mdA, mdB

def compute_mda(md, norm='manhattan', precision=0.2, threshold=0.2, area='simpson'):
    """ Compute accumulation between minimum distances.
        Gives the y axis, useful for privacy/resemblance metrics.

        :param md: Minimum distances of samples from distribution (already calculated for complexity reason)
        :param precision: discrepancy between two values on x axis
        :param threshold: privacy/resemblance trade-off for metrics. We want the minimum distances to be above this value for respect of privacy
        :param area: compute the area using the composite 'simpson' or 'trapezoidal' rule

        :return:
          (x, y): Coordinates of MDA curve for A
          (privacy, resemblance):
            privacy: area above the curve on the left side of the threshold. We want it to be maximal.
            resemblance: area under the curve on the right side of the threshold. We want it to be maximal.
          threshold: return the threshold for plot
    """
    mini, maxi = 0, max(max(md), 1) # min(md)

    if (threshold <= 0) or (threshold >= 1):
        print('Warning: threshold must be between 0 and 1.')

    # x axis
    x = np.arange(mini, maxi, precision)

    # y axis
    y = []

    for e in x:
        y.append(sum(1 for i in md if i < e))

    # Normalization
    x = normalize(x, normalization='min-max')
    y = normalize(y,normalization='min-max')
    precision = x[1] - x[0]
    #threshold = x[i]

    # Index of threshold in x
    i = int(np.ceil(threshold / precision))

    if area == 'simpson':
        compute_area = simps
    elif area == 'trapezoidal':
        compute_area = trapz
    else:
        raise ValueError('Argument area is invalid.')

    yl, yr = y[:i], y[i:]

    # Privacy: area under left curve
    privacy = 1 - (compute_area(yl, dx=precision) / threshold)

    # Resemblance: area under right curve
    resemblance = compute_area(yr, dx=precision) / (1 - threshold)

    return (x, y), (privacy, resemblance), threshold

def mmd(x, y):
    """ Compute the Maximum Mean Discrepancy Metric to compare empirical distributions.
        Ref: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. Journal of Machine Learning Research, 13(Mar), 723-773.
        :param x: Distribution
        :param y: Distribution
    """
    input_size = len(x[0])
    x, y = th.FloatTensor(x), th.FloatTensor(y)
    bandwiths = [0.01, 0.1, 1, 10, 100]
    s = th.cat([(th.ones([input_size, 1])).div(input_size),
                    (th.ones([input_size, 1])).div(-input_size)], 0)
    S = s.mm(s.t())
    S = Variable(S, requires_grad=False)
    X = th.cat([x, y], 0)
    # dot product between all combinations of rows in 'X'
    XX = X @ X.t()
    # dot product of rows with themselves
    # Old code : X2 = (X * X).sum(dim=1)
    X2 = XX.diag().unsqueeze(0)
    # exponent entries of the RBF kernel (without the sigma) for each
    # combination of the rows in 'X'
    # -0.5 * (i^Ti - 2*i^Tj + j^Tj)
    exponent = XX - 0.5 * (X2.expand_as(XX) + X2.t().expand_as(XX))
    lossMMD = th.sum(S * sum([(exponent * (1./bandwith)).exp() for bandwith in bandwiths]))
    return lossMMD.sqrt()

def chi_square(col1, col2):
    """ Performs Chi2 on two DataFrame columns

        :param col1: First column (variable)
        :param col2: Second column (variable)
        :return: Result of Chi2
    """
    return chi2_contingency(np.array([col1, col2]))

def kolmogorov_smirnov(col1, col2):
    """ Performs Kolmogorov-Smirnov test on two DataFrame columns

        :param col1: First column (variable)
        :param col2: Second column (variable)
        :return: Result of Kolmogorov-Smirnov test
    """
    res = ks_2samp(col1, col2)
    return res[0].round(3), res[1].round(3)

def kullback_leibler(freq1, freq2):
    """ Performs KL divergence on probability distributions
        Return a couple because this is not symetric

        :param freq1: Frequency distribution of the first variable.
        :param freq2: Frequency distribution of the second variable.
        :return: Kullback-Leibler divergence
    """
    return entropy(freq1, qk=freq2).round(3), entropy(freq2, qk=freq1).round(3)

def mutual_information(freq1, freq2):
    """ Performs the Kullback-Leibler divergence of the joint distribution with the product distribution of the marginals.

        :param freq1: Frequency/probability distribution of the first variable.
        :param freq2: Frequency/probability distribution of the second variable.
        :return: The score
    """
    return mutual_info_score(freq1, freq2).round(3)

def jensen_shannon(P, Q):
    """ Performs the Jensen-Shannon divergence on probability distributions.
        This metric is symetric.

        :param P: Frequency/probability distribution of the first variable.
        :param Q: Frequency/probability distribution of the second variable.
    """
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return (0.5 * (entropy(_P, _M) + entropy(_Q, _M))).round(3)

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
    """ Divergence based on (dist_to_nearest_miss - dist_to_nearest_hit)
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
    # TODO
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
