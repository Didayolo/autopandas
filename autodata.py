# AutoData
# Based on Pandas, numpy, sklearn, matplotlib

# TODO #####################################
# Find an example CSV with :
# Categorical, numerical and missing values

# Simplify AutoML (une méthode par affichage, traitement, etc)
# Le notebook appelle chaque méthode avec une explication

# For each method (display, etc.) parameter "key" gives the wanted subset of data
############################################

# Imports
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA

sys.path.append('utils')
#import utilities
import imputation
import encoding
import normalization


def read_csv(*args, **kwargs):
    return AutoData(pd.read_csv(*args, **kwargs))


def read_automl():
    pass


class AutoData(pd.DataFrame):
    """ AutoData is a data structure extending Pandas Dataframe.
        Its objective is to allow to quickly get to grips with a dataset.
    """

    _metadata = ['indexes']
    indexes = {'header':range(5)} # header, train, test, valid, X, y

    ## 1. #################### READ/WRITE DATA ######################

    # TODO
    # Read AutoML, CSV, TFRecords
    # Infer CSV separator
    # Init info, etc.

    def __init__(self, *args, **kwargs): # indexes = None
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.info = {}
        self.get_types() # find categorical and numerical variables


    @property
    def _constructor(self):
        return AutoData


    def to_automl(self):
        pass


    def set_index(self, key, value):
        self.indexes[key] = value


    def get_index(self, key=None):
        """ Return rows, columns
        """
        if key is None:
            rows = self.index
            columns = list(self)
            return rows, columns

        elif key in ['train', 'valid', 'test', 'header']:
            rows = self.indexes[key]
            columns = list(self)

        elif key in ['X', 'y', 'categorical', 'numerical']:
            rows = self.index
            columns = self.indexes[key]

        elif '_' in key:
            v, h = key.split('_')
            rows = self.indexes[h]
            columns = self.indexes[v]

        else:
            raise Exception('Unknown key.')

        return rows, columns


    def get_data(self, key=None):
        """ Get data
        """
        return self.loc[self.get_index(key)]


    def get_types(self):
        """ Compute variables types: Numeric or Categorical.
            This information is then stored as indexes with key 'numerical' and 'categorical'.
        """
        N = self.shape[0]
        prop = int(N / 25) # Arbitrary proportion of different values where a numerical variable is considered categorical

        categorical_index = []
        numerical_index = []

        for column in list(self):
            p = len(self[column].unique()) # number of unique values in column

            if (p <= prop) or any(isinstance(i, str) for i in self[column]):
                categorical_index.append(column)
            else:
                numerical_index.append(column)

        self.indexes['categorical'] = categorical_index
        self.indexes['numerical'] = numerical_index


    def get_task(self):
        """ TODO: multiclass?
        """
        if 'y' in self.indexes.keys():
            for c in self.indexes['y']:
                if c in self.indexes['numerical']:
                    return 'regression'

            else:
                return 'classification'

        else:
            raise Exception('No class is defined. Please use set_class method to define one.')

        def merge(data):
            pass

    ## 2. ###################### PROCESSINGS #########################
    # Imputation, encoding, normalization

    # TODO
    # Avoid data leakage during processing
    # train/test/valid split shuffle

    def train_test_split(self, test_size=0.3, shuffle=True, valid=False, valid_size=0.1):
        """ Procedure
            TODO shuffle
        """
        N = self.shape[0]
        split = round(N * (1 - test_size))
        self.set_index('train', range(split))
        self.set_index('test', range(split, N))


    def set_class(self, y):
        """ Procedure
            Define the column(s) representing a class (y).
        """
        X = list(self) # column names

        # y is 1 column
        if isinstance(y, str) or isinstance(y, int):
            self.set_index('y', [y])
            X.remove(y)

        # y is array-like
        else:
            self.set_index('y', y)
            for name in y:
                X.remove(name)

        self.set_index('X', X)


    def imputation(self, method='most', key=None):
        """ Impute missing values.
            :param method: None, 'remove', 'most', 'mean', 'median'
            :return: Data with imputed values.
            :rtype: AutoData
        """
        data = self.get_data()
        rows, columns = self.get_index(key)

        for column in columns:
            if method == 'remove':
                data = imputation.remove(data, column)

            elif method == 'most':
                data = imputation.most(data, column)

            elif method == 'mean':
                data = imputation.mean(data, column)

            elif method == 'median':
                data = imputation.median(data, column)

            else:
                raise Exception('Unknown imputation method: {}'.format(method))

        return data


    def normalization(self, method='standard', key=None):
        """ Normalize data.
            :param method: 'standard', 'min-max', None
            :return: Normalized data.
            :rtype: AutoData
        """
        data = self.get_data()
        rows, columns = self.get_index(key)

        for column in columns:
            if method == 'standard':
                data = normalization.standard(data, column)

                # TODO:
                #train, (mean, std) = normalization.standard(train, column, return_param=True)
                #test = normalization.standard(test, column, mean, std)

            elif method in ['min-max', 'minmax', 'min_max']:
                data = normalization.min_max(data, column)

            else:
                raise Exception('Unknown normalization method: {}'.format(method))

        return data


    def encoding(self, method='label', key=None):
        """ Encode categorical variables.
            :param method: 'none', 'label', 'one-hot', 'rare-one-hot', 'target', 'likelihood', 'count', 'probability'
            :param target: Target column name (target encoding).
            :param coeff: Coefficient defining rare values (rare one-hot encoding).
                          A rare category occurs less than the (average number of occurrence * coefficient).
        """
        data = self.get_data()
        rows, columns = self.get_index(key)

        for column in columns:
            data = encoding.label(data, column)

        return data


    ## 3. ###################### VISUALIZATION #######################

    # TODO

    def pca(self, verbose=False, **kwargs):
        """
            Compute PCA.
            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for PCA (see sklearn doc)
            :return: Tuple (pca, X) containing a PCA object (see sklearn doc) and the transformed data
            :rtype: Tuple
        """
        pca = PCA(**kwargs)
        X = pca.fit_transform(self)

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


    def show_pca():
        pass


    def plot(self, key=None, max_features=16):
        """ Show feature pairplots.
            TODO be able to pass column name ??
        """
        feat_num = self.shape[1]
        if feat_num < max_features: # TODO selection, plot with y
            sns.set(style="ticks")
            print('{} set plot'.format(key))
            data = self.get_data(key)
            sns.pairplot(data)
            plt.show()
        else:
            print('Too much features to pairplot. Number of features: {}, max features to plot set at: {}'.format(feat_num, max_features))


    ## 4. ######################## BENCHMARK ##########################

    # TODO
    # Score reports, confusion matrix

    def score(self, model=None, metric=None):
        """ Benchmark ...
        """
        if model is None:
            if self.get_task() == 'classification':
                model = RandomForestClassifier()

            elif self.get_task() == 'regression':
                model = RandomForestRegressor()

            else:
                raise 'Unknown task.'

        if 'test' not in self.indexes:
            raise Exception('No train/test split.')

        elif 'y' not in self.indexes:
            raise Exception('No class.')

        # Let's go!
        else:
            X_train = self.get_data('X_train')
            y_train = self.get_data('y_train')
            X_test = self.get_data('X_test')
            y_test = self.get_data('y_test')

            # mono-class
            if y_train.shape[1] == 1:
                y_train = y_train.values.ravel()
                y_test = y_test.values.ravel()

            model.fit(X_train, y_train)
            return model.score(X_test, y_test)
