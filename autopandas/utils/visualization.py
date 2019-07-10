# Plot Functions

import matplotlib.pyplot as plt
import seaborn as sns

def plot(data, key=None, ad=None, max_features=2, save=None, c=None, **kwargs):
    """ Show feature pairplots.
        TODO be able to pass column name ?
        Automatic selection ?
        :param ad: AutoData frame to plot in superposition
        :param save: filename to save fig if not None
        Class coloration only if y is categorical (classification)
    """
    data = data.get_data(key)
    feat_num = data.shape[1]
    if feat_num <= max_features:
        sns.set(style="ticks")
        if key is not None:
            print('{} set plot'.format(key))
        # if data.has_class
            # hue=y
            # TODO class coloration
        if feat_num == 2 and c is not None:
            # TEST
            # TODO
            plt.scatter(data[0], data[1], c=c, alpha=.4, s=3**2, cmap='viridis')
            plt.show()
        if ad is None:
            sns.pairplot(data, **kwargs)
        else: # if two dataframes to plot
            if feat_num == 2 and ad.shape[1] == 2: # if 2 features, overlay plots
                x1, y1, x2, y2 = data.iloc[:,0], data.iloc[:,1], ad.iloc[:,0], ad.iloc[:,1]
                plt.plot(x1, y1, 'o', alpha=.9, color='blue') #, label=label1) # lw=2, s=1, color='blue',
                plt.plot(x2, y2, 'x', alpha=.8, color='orange') #, marker='x') #, label=label2) # lw=2, s=1
                plt.axis([min(min(x1), min(x2)), max(max(x1), max(x2)), min(min(y1), min(y2)), max(max(y1), max(y2))])
            else:
                print('Overlay plot is only for 2 dimensional data.')
                sns.pairplot(data, **kwargs)
                plot(ad, key=key, max_features=max_features, palette='husl')
        if save is not None:
            plt.savefig(save)
        plt.show()
    else:
        #plt.matshow(data) # TODO
        sns.heatmap(data) # TODO
        #print('Too much features to pairplot. Number of features: {}, max features to plot set at: {}'.format(feat_num, max_features))

def heatmap(data, **kwargs):
    sns.heatmap(data, **kwargs)

def correlation(data, **kwargs):
    corr = data.corr()
    sns.heatmap(corr, **kwargs)

def compare_marginals(data1, data2, key=None, method='all', target=None, save=None, name1='dataset 1', name2='dataset2'):
    """ TODO
        Plot the metric for each variable from ds1 and ds2
        Mean, standard deviation or correlation with target.

        :param method: 'mean', 'std', 'corr', 'all'
        :param target: column name for the target for correlation method
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
            y2 = X2[X2.indexes['y'][0]] #.get_data('y') # TODO
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
