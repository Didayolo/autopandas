# Dimensionality reduction functions

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction import FeatureHasher
import pandas as pd
import autopandas
import numpy as np

def pca(data, key=None, verbose=False, **kwargs):
    """ Compute PCA.
        Use kwargs for additional PCA parameters (cf. sklearn doc).

        Parameters
        ----------
        key : Indexes key to select data.
        verbose : Display additional information during run.

        Returns
        -------
        AutoData
            Transformed data
    """
    pca = PCA(**kwargs)
    X = pca.fit_transform(data.get_data(key))
    if verbose:
        print('Explained variance ratio of the {} components: \n {}'.format(pca.n_components_,
                                                                            pca.explained_variance_ratio_))
        plt.bar(left=range(pca.n_components_),
                height=pca.explained_variance_ratio_,
                width=0.3,
                tick_label=range(pca.n_components_))
        plt.title('Explained variance ratio by principal component')
        plt.show()
    return X

def tsne(data, key=None, verbose=False, **kwargs):
    """ Compute T-SNE.
        Use kwargs for additional T-SNE parameters (cf. sklearn doc)

        Parameters
        ----------
        key : Indexes key to select data.
        verbose : Display additional information during run.

        Returns
        -------
        AutoData
            Transformed data
    """
    tsne = TSNE(**kwargs)
    X = tsne.fit_transform(data.get_data(key))
    return X

def lda(data, key=None, verbose=False, **kwargs):
    """ Compute Linear Discriminant Analysis.
        Use kwargs for additional LDA parameters (cf. sklearn doc)

        Parameters
        ----------
        key : Indexes key to select data.
        verbose : Display additional information during run.

        Returns
        -------
        AutoData
            Transformed data
    """
    lda = LinearDiscriminantAnalysis(**kwargs)
    X = data.get_data('X').get_data(key)
    y = data.get_data('y').get_data(key)
    if y.shape[1] > 1:
        print("WARNING: LDA can't handle multi-output class. Only the first column will be used.\nUse set_class method to define another target before calling lda.")
        y = y[y.columns[0]]
    X = lda.fit_transform(X, np.array(y).ravel()) # ravel to avoid warnings
    return X

def feature_hashing(data, key=None, n_features=10, **kwargs):
    """ Feature hashing.

        :param n_features: Wanted number of features after feature hashing.
    """
    data = data.get_data(key)
    h = FeatureHasher(n_features=n_features, **kwargs)
    data = data.to_dict('records')
    data = h.transform(data)
    data = data.toarray() # to array
    data = pd.DataFrame(data) # to df
    return data
