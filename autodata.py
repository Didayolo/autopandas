# AutoData
# Based on pandas, numpy, sklearn, matplotlib and seaborn

# TODO #####################################
# Documentation
# :param key: on documentation
# Add explanations in the notebook

# For each method (display, etc.) parameter "key" gives the wanted subset of data
############################################

# Imports
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

# Import project files
sys.path.append('utils')
import imputation
import encoding
import normalization
import reduction
import visualization
import benchmark
import metric


def read_csv(*args, **kwargs):
    # Infer CSV separator: sep=None, engine='python'
    return AutoData(pd.read_csv(*args, **kwargs), reset=True)


def read_automl(input_dir, basename):
    """ Read files in AutoML format.
        TODO
    """

    feat_name_file = os.path.join(input_dir, basename + '_feat.name')
    feat_name = pd.read_csv(feat_name_file, header=None).values.ravel() if os.path.exists(feat_name_file) else None

    label_name_file = os.path.join(input_dir, basename + '_label.name')
    label_name = pd.read_csv(label_name_file, header=None).values.ravel() if os.path.exists(label_name_file) else None

    # if exists
    if os.path.exists(os.path.join(input_dir, basename + '.data')):

        # read .data and .solution
        pd.read_csv(filepath, sep=' ', header=None)

    # create AutoData object
    data = AutoData(df, reset=True)

    # class ?
    data.set_class()

    # train/valid/test ?
    data.indexes['train'] = [0]


class AutoData(pd.DataFrame):
    """ AutoData is a data structure extending Pandas Dataframe.
        The goal is to quickly get to grips with a dataset.
    """

    _metadata = ['indexes']
    indexes = {'header':range(5)} # header, train, test, valid, X, y

    ## 1. #################### READ/WRITE DATA ######################

    # TODO
    # Read AutoML, TFRecords
    # Init/save info, etc. (AutoML info)

    def __init__(self, *args, reset=False, **kwargs): # indexes = None
        pd.DataFrame.__init__(self, *args, **kwargs)
        # self.info = {}

        #self.indexes = {'header':range(5)}

        if 'numerical' not in self.indexes.keys() or reset:
            # find categorical and numerical variables
            self.get_types()


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

        elif key in ['train', 'valid', 'test', 'header']:
            rows = self.indexes[key]
            columns = list(self) # all features

        elif key in ['X', 'y', 'categorical', 'numerical']:
            rows = self.index # all rows
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


    def add(self, data):
        """ Combine rows of two AutoData objects: self and data.
        """
        return pd.concat([self, data])


    def merge(self, data):
        """ Same indexes but data is a modified part of self.
        """
        pass


    ## 2. ###################### PROCESSINGS #########################
    # Imputation, encoding, normalization

    # TODO
    # Avoid data leakage during processing
    # train/test/valid split shuffle
    # Dimensionality reduction (method) (only on X)

    def train_test_split(self, test_size=0.3, shuffle=True, valid=False, valid_size=0.1):
        """ Procedure
            TODO shuffle
        """
        N = self.shape[0]
        split = round(N * (1 - test_size))

        train_index = range(split)
        valid_index = []
        test_index = range(split, N-1)

        self.set_index('train', train_index)
        self.set_index('valid', valid_index)
        self.set_index('test', test_index)


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


    def pca(self, key=None, verbose=False, **kwargs):
        """
            Compute PCA.
            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for PCA (see sklearn doc)
            :return: Transformed data
            :rtype: AutoData
        """
        return AutoData(reduction.pca(self, key=key, verbose=verbose, **kwargs)) #, reset=True)


    def tsne(self, key=None, verbose=False, **kwargs):
        """
            Compute T-SNE.
            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for T-SNE (see sklearn doc)
            :return: Transformed data
            :rtype: AutoData
        """
        return AutoData(reduction.tsne(self, key=key, verbose=verbose, **kwargs)) #, reset=True)


    def lda(self, key=None, verbose=False, **kwargs):
        """
            Compute Linear Discriminant Analysis.
            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for LDA (see sklearn doc)
            :return: Transformed data
            :rtype: AutoData
        """
        return AutoData(reduction.lda(self, key=key, verbose=verbose, **kwargs)) #, reset=True)


    def reduction(self, method='pca', key=None, verbose=False, **kwargs):
        """ Dimensionality reduction
            method: pca, lda, tsne
        """
        data = self.get_data(key)

        if method == 'pca':
            data = data.pca(key=key, verbose=verbose, **kwargs)

        elif method == 'tsne':
            data = data.tsne(key=key, verbose=verbose, **kwargs)

        elif method == 'lda':
            data = data.lda(key=key, verbose=verbose, **kwargs)

        else:
            raise Exception('Unknown dimensionality reduction method: {}'.format(method))

        return data


    ## 3. ###################### VISUALIZATION #######################

    # TODO
    # Class coloration on plots!


    def plot(self, key=None, max_features=12, **kwargs):
        """ Show feature pairplots.
            TODO be able to pass column name ?
            Automatic selection ?
        """
        visualization.plot(self, key=key, max_features=max_features, **kwargs)


    def plot_pca(self, key):
        self.pca(key, n_components=2).plot()


    def heatmap(self, **kwargs):
        visualization.heatmap(self, **kwargs)


    def correlation(self, **kwargs):
        visualization.correlation(self, **kwargs)


    ## 4. ######################## BENCHMARK ##########################

    # TODO
    # Different scoring metrics
    # Score reports, confusion matrix
    # Model selection
    # Model tuning

    def score(self, model=None, metric=None, method='baseline'):
        """ Benchmark ...
        """
        return benchmark.score(self, model=model, metric=metric, method=method)


    # Distribution comparator
    def distance(self, data, method=None):
        """ Distance between two AutoData frames.
            There is a lot of methods to add (cf. utilities.py and metric.py)
        """
        return metric.nn_discrepancy(self, data)


    #################### ALIAS #######################
    # How to automate this ?
    # __getattr__ ?
    # Need to think about this
    #X(), y(), X_train, y_test, numerical, header, etc.
    # set_X ?
    ##################################################
