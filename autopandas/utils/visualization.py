# Plot Functions

from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Hierarchical clustering
import matplotlib as mpl
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import string
import time
import sys, os
import getopt

def plot(data, key=None, ad=None, c=None, save=None, names=None, cmap='viridis', **kwargs):
    """ Plot AutoData frame.
        * Distribution plot for 1D data
        * Scatter plot for 2D data
        * Heatmap for >2D data

        For scatter plot, coloration is by default the class if possible, or can be defined with c parameter.

        :param key: Key for subset selection (e.g. 'X_train' or 'categorical')
        :param ad: AutoData frame to plot in superposition
        :param c: Sequence of color specifications of length n (e.g. data.get_data('y'))
        :param save: Path/filename to save figure (if not None)
    """
    if c is None:
        cmap = None
    if names is None:
        names = ['data 1', 'data 2']
    data = data.get_data(key)
    feat_num = data.shape[1]
    sns.set(style="ticks")
    if key is not None:
        print('{} set plot'.format(key))
    if ad is None: # Only one dataframe to plot
        if feat_num == 1: # Dist plot
            pairplot(data, save=save, **kwargs)
        elif feat_num == 2: # 2D plot
            if data.has_class() and c is None: # use class for coloration
                c = data.get_data('y')
            title = None
            if isinstance(c, pd.DataFrame): # c has to be a 1D sequence
                if len(c.columns) > 1:
                    warn('Only the first column will be used for coloration.')
                title = c.columns[0]
                c = list(c[title])
            fig, ax = plt.subplots()
            scatter = ax.scatter(data[data.columns[0]], data[data.columns[1]], c=c, alpha=.4, s=3**2, cmap=cmap)
            legend = ax.legend(*scatter.legend_elements(), loc='center left', bbox_to_anchor=(1, 0.5), title=title)
            plt.show()
        else: # Not 2D plot
            heatmap(data, save=save, **kwargs)
    else: # 2 dataframes to plot
        if feat_num == 1 and ad.shape[1] == 1: # 1D distributions
            plt.scatter(data, ad) # plot together, or overlay distplots
            warn('TODO: legend and all.')
        elif feat_num == 2 and ad.shape[1] == 2: # if 2 features, overlay plots
            x1, y1, x2, y2 = data.iloc[:,0], data.iloc[:,1], ad.iloc[:,0], ad.iloc[:,1]
            plt.plot(x1, y1, 'o', alpha=.9, color='blue', label=names[0]) # lw=2, s=1, color='blue',
            plt.plot(x2, y2, 'x', alpha=.8, color='orange', label=names[1]) #, marker='x') # lw=2, s=1
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.axis([min(min(x1), min(x2)), max(max(x1), max(x2)), min(min(y1), min(y2)), max(max(y1), max(y2))])
        else: # Not 2D plots
            print('Overlay plot is only for 1D or 2D data.')
            heatmap(data, save=save, **kwargs)
            plot(ad, save='bis_'+save, palette='husl')
    if save is not None:
        plt.savefig(save)
    plt.show()

def pairplot(data, key=None, max_features=12, force=False, save=None, **kwargs):
    """ Plot pairwise relationships between features.

        :param key: Key for subset selection (e.g. 'X_train' or 'categorical')
        :param max_features: Max number of features to pairplot.
        :param force: If True, plot the graphs even if the number of features is grater than max_features.
        :param save: Path/filename to save figure (if not None)
    """
    data = data.get_data(key)
    feat_num = data.shape[1]
    if (feat_num <= max_features) or force==True:
        f = sns.pairplot(data, **kwargs)
        if save is not None:
            f.savefig(save)
    else:
        print('Max number of features to pairplot is set to {} and your data has {} features.\nIncrease max_features or set force to True to proceed.'.format(max_features, feat_num))

def heatmap(data, key=None, save=None, **kwargs):
    """ Plot data heatmap.

        :param key: Key for subset selection (e.g. 'X_train' or 'categorical')
        :param save: Path/filename to save figure (if not None)
    """
    data = data.get_data(key)
    f = sns.heatmap(data, **kwargs)
    if save is not None:
        f.savefig(save)

def correlation(data, key=None, save=None, **kwargs):
    """ Plot correlation matrix.

        :param key: Key for subset selection (e.g. 'X_train' or 'categorical')
        :param save: Path/filename to save figure (if not None)
    """
    data = data.get_data(key)
    corr = data.corr()
    f = sns.heatmap(corr, **kwargs)
    if save is not None:
        f.savefig(save)

def compare_marginals(data1, data2, key=None, method='all', target=None, save=None, name1='dataset 1', name2='dataset2'):
    """ Plot the metric (e.g. mean) for each variable from data1 and data2.
        If the distributions are similar, the points will follow the y=x line.
        Mean, standard deviation or correlation with target.
        data1 and data2 has to have the same number of features.

        :param method: 'mean', 'std', 'corr', 'all'
        :param target: Column name for the target for correlation method
        :param save: Path to save the figure (doesn't save if 'save' is None).
    """
    has_class = data1.has_class() and data2.has_class()
    X1 = data1.get_data(key)
    X2 = data2.get_data(key)
    x_mean, y_mean = [], []
    x_std, y_std = [], []
    x_corr, y_corr = [], []
    if method in ['mean', 'all']:
        for column in list(X1.columns):
            x_mean.append(X1[column].mean())
            y_mean.append(X2[column].mean())
    if method in ['std', 'all']:
        for column in list(X1.columns):
            x_std.append(X1[column].std())
            y_std.append(X2[column].std())
    skip_corr = False # skip correlation plot if no target and method=='all'
    if method in ['corr', 'correlation', 'all']:
        if target is None: # no defined target
            if has_class:
                y1 = X1[X1.indexes['y'][0]] #.get_data('y') # TODO
                y2 = X2[X2.indexes['y'][0]] #.get_data('y')
            else: # no class and no target
                if method in ['corr', 'correlation']:
                    raise Excpetion('Cannot compute correlation with target. Please define a target column with target argument or define a class with set_class method.')
                else:
                    warn('Skipping "correlation with target" metric because there is no defined target.')
                    skip_corr = True
        else:
            y1 = X1[target]
            y2 = X2[target]
        if not skip_corr:
            # Flatten one-hot (dirty)
            if len(y1.shape) > 1:
                if y1.shape[1] > 1:
                    y1 = np.where(y1==1)[1]
                    y1 = pd.Series(y1)
            if len(y2.shape) > 1:
                if y2.shape[1] > 1:
                    y2 = np.where(y2==1)[1]
                    y2 = pd.Series(y2)
            for column in list(X1.columns):
                x_corr.append(X1[column].corr(y1))
                y_corr.append(X2[column].corr(y2))
    if method not in ['mean', 'std', 'corr', 'correlation', 'all']:
        raise Exception('{} metric is not taken in charge'.format(method))
    # Let's go
    if method == 'mean':
        plt.plot(x_mean, y_mean, 'o', color='b')
        plt.xlabel('Mean of variables in ' + name1)
        plt.ylabel('Mean of variables in ' + name2)
        plt.plot([0, 1], [0, 1], color='grey', alpha=0.4)
    elif method == 'std':
        plt.plot(x_std, y_std, 'o', color='g')
        plt.xlabel('Standard deviation of variables in ' + name1)
        plt.ylabel('Standard deviation of variables in ' + name2)
        plt.plot([0, 0.4], [0, 0.4], color='grey', alpha=0.4)
    elif method in ['corr', 'correlation']:
        plt.plot(x_corr, y_corr, 'o', color='r')
        plt.xlabel('Correlation with target of variables in ' + name1)
        plt.ylabel('Correlation with target of variables in ' + name2)
        plt.plot([-1, 1], [-1, 1], color='grey', alpha=0.4)
    elif method == 'all':
        plt.plot(x_mean, y_mean, 'o', color='b', alpha=0.9, label='Mean')
        plt.plot(x_std, y_std, 'o', color='g', alpha=0.8, label='Standard deviation')
        if not skip_corr:
            plt.plot(x_corr, y_corr, 'o', color='r', alpha=0.7, label='Correlation with target')
        plt.xlabel(name1 + ' variables')
        plt.ylabel(name2 +' variables')
        plt.legend(loc='upper left')
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)
        plt.plot([-1, 1], [-1, 1], color='grey', alpha=0.4)
    else:
        raise Exception('{} metric is not taken in charge'.format(method))
    if save is not None:
        plt.savefig(save)
    plt.show()

# hierarchical clustering heatmap
# need cleaning
def hierarchical_clustering(X, row_method='average', column_method='single',
                                    row_metric='euclidean', column_metric='euclidean',
                                    color_gradient='coolwarm'):
    """
    Show heatmap hierarchical clustering of X.
    This below code is based in large part on the protype methods:
    http://old.nabble.com/How-to-plot-heatmap-with-matplotlib--td32534593.html
    http://stackoverflow.com/questions/7664826/how-to-get-flat-clustering-corresponding-to-color-clusters-in-the-dendrogram-cre

    X is an (m by n) np.ndarray, m observations, n genes.
    """
    print(
        "\nPerforming hierarchical clustering using {} for columns and {} for rows".
        format(column_metric, row_metric))
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
