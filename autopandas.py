# AutoData
# Based on pandas, numpy, sklearn, auto-sklearn, matplotlib and seaborn

# TODO #####################################
# Documentation (on code and notebook)
# Clean other modules
# Check todo-lists below
############################################

# Imports
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

# Import project files
from .utils import imputation as imputation
from .utils import encoding as encoding
from .utils import normalization as normalization
from .utils import reduction as reduction
from .utils import visualization as visualization
from .utils import benchmark as benchmark
from .utils import metric as metric

# generators
from .generators import generators as generators

def read_csv(*args, **kwargs):
    """ Read data from CSV file.
        Default behaviour is to infer separator.
    """
    if ('sep' in kwargs) or ('engine' in kwargs):
        data = pd.read_csv(*args, **kwargs)
    else: # Infer CSV separator
        data = pd.read_csv(*args, **kwargs, sep=None, engine='python')
    return AutoData(data)

def read_automl(input_dir, basename):
    """ Read files in AutoML format.
        TODO
    """
    pass
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
    data = AutoData(df)

    # class ?
    data.set_class()

    # train/valid/test ?
    data.indexes['train'] = [0]
    """

def from_train_test(train, test):
    """ Create an AutoDataFrame from a train and a test DataFrame
    """
    ad = AutoData.append(train, test)
    ad.set_index('train', range(0, len(train)))
    ad.set_index('test', range(len(train), len(ad)))
    return ad

def plot(ad1, ad2, **kwargs):
    """ Alias for double plot.
    """
    ad1.plot(ad=ad2, **kwargs)

def compare_marginals(ad1, ad2, **kwargs):
    """ Alias for marginals comparison plot.
    """
    ad1.compare_marginals(ad=ad2, **kwargs)


class AutoData(pd.DataFrame):
    """ AutoData is a data structure extending Pandas Dataframe.
        The goal is to quickly get to grips with a dataset.
    """
    _metadata = ['indexes']

    ## 1. #################### READ/WRITE DATA ######################

    # TODO
    # Read AutoML, TFRecords
    # Init/save info, etc. (AutoML info)

    def __init__(self, *args, indexes=None, **kwargs):  # indexes = None

        pd.DataFrame.__init__(self, *args, **kwargs)
        # self.info = {} # maybe for later

        # indexes ('X', 'y', 'train', 'categorical', 'y_test', etc.)
        self.indexes = {'header': range(5)} if indexes is None else indexes

        # find categorical and numerical variables
        if 'numerical' not in self.indexes.keys():
            self.get_types()

        # train test split
        if 'train' not in self.indexes.keys():
            self.train_test_split()

    @property
    def _constructor(self):
        return AutoData

    def copy(self):
        """ Redefining copy to keep indexes from one copy to another.
        """
        data = pd.DataFrame.copy(self)
        data.indexes = self.indexes.copy()
        return data

    def to_csv(self, *args, **kwargs):
        """ Write data into a CSV file.
            Index is not written by default.
        """
        if ('index' in kwargs):
            super(AutoData, self).to_csv(*args, **kwargs)
        else: # Do not write index column
            super(AutoData, self).to_csv(*args, **kwargs, index=False)

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
            columns = list(self)  # all features

        elif key in ['X', 'y', 'categorical', 'numerical']:
            rows = self.index  # all rows
            columns = self.indexes[key]

        elif '_' in key:
            v, h = key.split('_')
            rows = self.indexes[h]
            columns = self.indexes[v]

        else:
            raise Exception('Unknown key.')

        return rows, columns

    def flush_index(self, key=None):
        """ Delete useless indexes for a specific set.
            For example:
              key='X_train'
              -> delete 'test' and 'y' indexes

            Maybe useless.
        """
        pass

    def get_data(self, key=None):
        """ Get data.
            :param key: wanted subset of data ('train', 'categorical_header', 'y', etc.)
        """
        data = self.loc[self.get_index(key)]
        data.indexes = self.indexes
        #data.flush_index(key)
        return data

    def get_types(self):
        """ Compute variables types: Numeric or Categorical.
            This information is then stored as indexes with key 'numerical' and 'categorical'.
        """
        N = self.shape[0]
        prop = int(N / 25)  # Arbitrary proportion of different values where a numerical variable is considered categorical

        categorical_index = []
        numerical_index = []

        for column in list(self):
            p = len(self[column].unique())  # number of unique values in column

            if (p <= prop) or any(isinstance(i, str) for i in self[column]):
                categorical_index.append(column)
            else:
                numerical_index.append(column)

        self.indexes['categorical'] = categorical_index
        self.indexes['numerical'] = numerical_index

    def get_task(self):
        """ TODO: multiclass?
        """
        if self.has_class():
            for c in self.indexes['y']:
                if c in self.indexes['numerical']:
                    return 'regression'

            else:
                return 'classification'

        else:
            raise Exception('No class is defined. Please use set_class method to define one.')

    # memo
    # to concat df by columns: join
    # to concat df by rows: append

    def merge(self, data):
        """ Same indexes but data is a modified part of self.
            Useless?
        """
        pass

    ## 2. ###################### PROCESSINGS #########################
    # Imputation, encoding, normalization

    # TODO
    # Avoid data leakage during processing
    # train/test/valid split shuffle
    # Dimensionality reduction (method) (only on X)

    def train_test_split(self,
                         test_size=0.3,
                         shuffle=True,
                         valid=False,
                         valid_size=0.1):
        """ Procedure
            TODO: shuffle
        """
        N = self.shape[0]
        split = round(N * (1 - test_size))

        train_index = range(split)
        valid_index = []
        test_index = range(split, N - 1)

        self.set_index('train', train_index)
        self.set_index('valid', valid_index)
        self.set_index('test', test_index)

    def set_class(self, y=None):
        """ Procedure
            Define the column(s) representing a class (y).
        """
        X = list(self)  # column names

        # no class (re-initialize)
        if y is None:
            self.set_index('y', [])

        # y is 1 column
        elif isinstance(y, str) or isinstance(y, int):
            self.set_index('y', [y])
            try:
                X.remove(y)
            except:
                raise Exception('Column "{}" does not exist.'.format(y))

        # y is array-like
        else:
            self.set_index('y', y)
            for name in y:
                try:
                    X.remove(name)
                except:
                    raise Exception('Column "{}" does not exist.'.format(name))
        self.set_index('X', X)

    def has_class(self):
        """ Return True if 'y' is defined and corresponds to one column (or more)
        """
        if ('y' in self.indexes.keys()) and (self.indexes['y'] != []):
                return True
        return False

    def imputation(self, method='most', key=None):
        """ Impute missing values.
            :param method: None, 'remove', 'most', 'mean', 'median'
            :return: Data with imputed values.
            :rtype: AutoData
        """
        data = self.copy()
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
        data = self.copy()
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
        data = self.copy()
        rows, columns = self.get_index(key)

        for column in columns:
            data = encoding.label(data, column)

        return data

    def pca(self, key=None, verbose=False, **kwargs):
        """ Compute PCA.
            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for PCA (see sklearn doc)
            :return: Transformed data
            :rtype: AutoData
        """
        data = self.copy()
        rows, columns = data.get_index(key)
        # compute PCA and copy indexes
        data = AutoData(
            reduction.pca(data, key=key, verbose=verbose, **kwargs),
            indexes=data.indexes)
        # variable are now only numerical
        data.indexes['categorical'] = []
        data.indexes['numerical'] = list(data)
        return data

    def tsne(self, key=None, verbose=False, **kwargs):
        """ Compute T-SNE.
            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for T-SNE (see sklearn doc)
            :return: Transformed data
            :rtype: AutoData
        """
        return AutoData(
            reduction.tsne(self, key=key, verbose=verbose, **kwargs))

    def lda(self, key=None, verbose=False, **kwargs):
        """
            Compute Linear Discriminant Analysis.
            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for LDA (see sklearn doc)
            :return: Transformed data
            :rtype: AutoData
        """
        return AutoData(
            reduction.lda(self, key=key, verbose=verbose, **kwargs))

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

    def plot(self, key=None, ad=None, max_features=12, save=None, **kwargs):
        """ Show feature pairplots.
            TODO be able to pass column name ?
            Automatic selection ?
        """
        visualization.plot(
            self,
            key=key,
            ad=ad,
            max_features=max_features,
            save=save,
            **kwargs)

    def plot_pca(self, key):
        self.pca(key, n_components=2).plot()

    def heatmap(self, **kwargs):
        visualization.heatmap(self, **kwargs)

    def correlation(self, **kwargs):
        visualization.correlation(self, **kwargs)

    def compare_marginals(self, ad, **kwargs):
        visualization.compare_marginals(self, ad, **kwargs)

    ## 4. ######################## BENCHMARK ##########################

    # TODO
    # Different scoring metrics
    # Score reports, confusion matrix
    # Model selection
    # Model tuning

    def score(self, model=None, metric=None, method='baseline', fit=True):
        """ Benchmark ...
        """
        return benchmark.score(
            self, model=model, metric=metric, method=method, fit=fit)

    # Distribution comparator
    def distance(self, data, method=None):
        """ Distance between two AutoData frames.
            There is a lot of methods to add (cf. utilities.py and metric.py)
            Usage example: ad1.distance(ad2, method='privacy')
        """
        if method is None:
            return metric.nn_discrepancy(self, data)
        elif method == 'adversarial_accuracy':
            return metric.adversarial_accuracy(self.get_data('train'), self.get_data('test'), data)
        else:
            raise Exception('Unknown distance metric: {}'.format(method))

    #################### ALIAS #######################
    # How to automate this ?
    # __getattr__ ?
    # Need to think about this
    #X(), y(), X_train, y_test, numerical, header, etc.
    # set_X ?
    ##################################################
