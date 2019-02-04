# Plot Functions

import matplotlib.pyplot as plt
import seaborn as sns


def plot(data, key=None, ad=None, max_features=12, save=None, **kwargs):
    """ Show feature pairplots.
        TODO be able to pass column name ?
        Automatic selection ?
        :param ad: AutoData frame to plot in superposition
        :param save: filename to save fig if not None
        Class coloration only if y is categorical (classification)
    """
    data = data.get_data(key)
    feat_num = data.shape[1]

    if feat_num < max_features:
        sns.set(style="ticks")

        if key is not None:
            print('{} set plot'.format(key))

        if 'y' in data.indexes:
            print('TODO: class coloration')
            # hue=y

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
        print('Too much features to pairplot. Number of features: {}, max features to plot set at: {}'.format(feat_num, max_features))


def heatmap(data, **kwargs):
    sns.heatmap(data, **kwargs)


def correlation(data, **kwargs):
    corr = data.corr()
    sns.heatmap(corr, **kwargs)
