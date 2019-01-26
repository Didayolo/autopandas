# Plot Functions

import matplotlib.pyplot as plt
import seaborn as sns


def plot(data, key=None, max_features=12, **kwargs):
    """ Show feature pairplots.
        TODO be able to pass column name ?
        Automatic selection ?
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

        sns.pairplot(data, **kwargs)
        plt.show()

    else:
        print('Too much features to pairplot. Number of features: {}, max features to plot set at: {}'.format(feat_num, max_features))


def heatmap(data, **kwargs):
    sns.heatmap(data, **kwargs)


def correlation(data, **kwargs):
    corr = data.corr()
    sns.heatmap(corr, **kwargs)
