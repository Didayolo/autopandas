# Plot Functions

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot(data, key=None, ad=None, c=None, save=None, **kwargs):
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
                    print('WARNING: only the first column will be used for coloration.')
                title = c.columns[0]
                c = list(c[title])
            fig, ax = plt.subplots()
            scatter = ax.scatter(data[data.columns[0]], data[data.columns[1]], c=c, alpha=.4, s=3**2, cmap='viridis')
            legend = ax.legend(*scatter.legend_elements(), loc='center left', bbox_to_anchor=(1, 0.5), title=title)
            plt.show()
        else: # Not 2D plot
            heatmap(data, save=save, **kwargs)
    else: # 2 dataframes to plot
        if feat_num == 1 and ad.shape[1] == 1: # 1D distributions
            plt.scatter(data, ad) # plot together, or overlay distplots
            print('WARNING: TODO legend and all.')
        elif feat_num == 2 and ad.shape[1] == 2: # if 2 features, overlay plots
            x1, y1, x2, y2 = data.iloc[:,0], data.iloc[:,1], ad.iloc[:,0], ad.iloc[:,1]
            plt.plot(x1, y1, 'o', alpha=.9, color='blue') #, label=label1) # lw=2, s=1, color='blue',
            plt.plot(x2, y2, 'x', alpha=.8, color='orange') #, marker='x') #, label=label2) # lw=2, s=1
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
    if (method == 'all' or method == 'corr') and (target is None and not has_class):
        raise OSError('You must define a target to use {} method.'.format(method))
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
    if method in ['corr', 'all']:
        if has_class and (target is None):
            y1 = X1[X1.indexes['y'][0]] #.get_data('y') # TODO
            y2 = X2[X2.indexes['y'][0]] #.get_data('y')
        else:
            y1 = X1[target]
            y2 = X2[target]
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
    elif method not in ['mean', 'std', 'corr', 'all']:
        raise OSError('{} metric is not taken in charge'.format(method))
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
    elif method == 'corr':
        plt.plot(x_corr, y_corr, 'o', color='r')
        plt.xlabel('Correlation with target of variables in ' + name1)
        plt.ylabel('Correlation with target of variables in ' + name2)
        plt.plot([-1, 1], [-1, 1], color='grey', alpha=0.4)
    elif method == 'all':
        plt.plot(x_mean, y_mean, 'o', color='b', alpha=0.9, label='Mean')
        plt.plot(x_std, y_std, 'o', color='g', alpha=0.8, label='Standard deviation')
        plt.plot(x_corr, y_corr, 'o', color='r', alpha=0.7, label='Correlation with target')
        plt.xlabel(name1 + ' variables')
        plt.ylabel(name2 +' variables')
        plt.legend(loc='upper left')
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)
        plt.plot([-1, 1], [-1, 1], color='grey', alpha=0.4)
    else:
        raise OSError('{} metric is not taken in charge'.format(method))
    if save is not None:
        plt.savefig(save)
    plt.show()
