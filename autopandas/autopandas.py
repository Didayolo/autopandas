# AutoPandas
# Based on pandas, numpy, sklearn, auto-sklearn, matplotlib and seaborn

# TODO #####################################
# Documentation (on code and notebook)
# Clean other modules
############################################

# Imports
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import pandas as pd
import random
import numpy as np

# Import project files
from .utils import imputation as imputation
from .utils import encoding as encoding
from .utils import normalization as normalization
from .utils import reduction as reduction
from .utils import visualization as visualization
from .utils import benchmark as benchmark
from .utils import metric as metric
from .utils import automl as automl
from .utils import sdv as sdv
# generators
from .generators import *
# some transformations
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection

def read_csv(*args, **kwargs):
    """ Read data from CSV file.
        Default behaviour is to infer the separator.
        It then creates an AutoData object, so the numerical/categorical inference and train/test split are done automatically.
    """
    if ('sep' in kwargs) or ('engine' in kwargs):
        data = pd.read_csv(*args, **kwargs)
    else: # Infer CSV separator
        data = pd.read_csv(*args, **kwargs, sep=None, engine='python')
    return AutoData(data)

# memo
# to concat df by columns: join
# to concat df by rows: append
######

def from_train_test(train, test):
    """ Create an AutoData frame from a train and a test DataFrames.
    """
    # Cast if needed
    if not isinstance(train, pd.DataFrame):
        train = AutoData(train)
    if not isinstance(test, pd.DataFrame):
        test = AutoData(test)
    # Concatenate by rows
    indexes = train.indexes.copy()
    ad = AutoData.append(train, test)
    ad.indexes = indexes
    ad.set_indexes('train', range(0, len(train)))
    ad.set_indexes('test', range(len(train), len(ad)))
    ad = ad.reset_index(drop=True)
    return ad

def from_X_y(X, y):
    """ Create an AutoData frame from a X and a y (class) DataFrames.
    """
    # Cast if needed
    if not isinstance(X, pd.DataFrame):
        X = AutoData(X)
    if not isinstance(y, pd.DataFrame):
        y = AutoData(y)
    # Concatenate by columns
    X.columns = X.columns.astype(str)
    y.columns = y.columns.astype(str)
    ad = AutoData.join(X, y, lsuffix='_X', rsuffix='_y')
    # Update indexes
    X_index = [x+'_X' for x in X.columns if x in y.columns] + [x for x in X.columns if x not in y.columns]
    y_index = [y+'_y' for y in y.columns if y in X.columns] + [y for y in y.columns if y not in X.columns]
    ad.set_indexes('X', X_index)
    ad.set_indexes('y', y_index)
    return ad

def plot(ad1, ad2, **kwargs):
    """ Alias for double plot.
    """
    ad1.plot(ad=ad2, **kwargs)

def compare_marginals(ad1, ad2, **kwargs):
    """ Alias for marginals comparison plot.
    """
    ad1.compare_marginals(ad=ad2, **kwargs)

def distance(ad1, ad2, method=None): #, **kwargs): TODO
    """ Alias for distance between AutoData.
    """
    return ad1.distance(ad2, method=method) #, **kwargs)

class AutoData(pd.DataFrame):
    _metadata = ['indexes'] # Python magic

    ## 1. #################### READ/WRITE DATA ######################

    # TODO
    # Read AutoML, TFRecords
    # Init/save info, etc. (AutoML info)

    def __init__(self, *args, indexes=None, **kwargs):  # indexes = None
        """ AutoData is a data structure extending Pandas DataFrame.
            The goal is to quickly get to grips with a dataset.
            An AutoData object represents a 2D data frame with:
              - Examples in rows
              - Features in columns

            If needed, automatically do:
            * numerical/categorical variables inference
            * train/test split
        """
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
        """ Re-defines copy to keep indexes from one copy to another.
        """
        data = pd.DataFrame.copy(self)
        data.indexes = self.indexes.copy()
        return data

    def __call__(self, *args, **kwargs):
        """ Calling the AutoData object is equivalent to call the get_data method.
            e.g. data('y') is an alias for data.get_data('y').
        """
        return self.get_data(*args, **kwargs)

    def to_csv(self, *args, **kwargs):
        """ Write data into a CSV file.
            Index is not written by default.
        """
        if ('index' in kwargs):
            super(AutoData, self).to_csv(*args, **kwargs)
        else: # Do not write index column
            super(AutoData, self).to_csv(*args, **kwargs, index=False)

    def to_automl(self, path='.', name='autodata'):
        """ Write files in AutoML format.
            AutoML format is ideal to create a Codalab competition.

            :param path: where to save the dataset
            :param name: name of the dataset to put into filenames
        """
        automl.to_automl(self, path, name=name)

    def set_indexes(self, key, value):
        """ Set an entry in the index.

            For example: data.set_indexes('y', 'income')
        """
        self.indexes[key] = value

    def get_index(self, key=None):
        """ Return rows, columns.
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

    def flush_index(self, key=None, compute_types=True):
        """ Delete non-existing columns from indexes.

            #Delete useless indexes for a specific set.
            #For example:
            #  key='X_train'
            #  -> delete 'test' and 'y' indexes
            #Maybe useless.
        """
        # Rows
        # Delete from indexes non-existing rows
        # Columns
        for k in ['categorical', 'numerical', 'X', 'y']:
            if k in self.indexes:
                self.indexes[k] = [x for x in self.indexes[k] if x in self.columns]
        if compute_types:
            self.get_types()

    def get_data(self, key=None):
        """ Get data.

            :param key: wanted subset of data ('train', 'categorical_header', 'y', etc.)
        """
        data = self.loc[self.get_index(key)].copy()
        data.flush_index(compute_types=False) # save time by not re-computing variable types
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
        """ Return 'regression' or 'classification' regarding the target type.
            TODO: multiclass?
        """
        if self.has_class():
            for c in self.indexes['y']:
                if c in self.indexes['numerical']:
                    return 'regression'
            else:
                return 'classification'
        else:
            raise Exception('No class is defined. Please use set_class method to define one.')

    ##################################################################
    # DESCRIPTORS
    def task(self):
        """ Alias for get_task method.
            Return 'regression' or 'classification'.
        """
        return self.get_task()

    def input_size(self):
        """ Number of features in X.
        """
        if self.has_class():
            return self('X').shape[1]
        else:
            return self.shape[1]

    def output_size(self):
        """ Number of unique classes.
        """
        if self.has_class():
            nunique = self('y').nunique()
            if len(self.indexes['y']) > 1: # multi-class
                return list(nunique)
            else:
                return int(nunique)
        else:
            raise Exception('No target defined. Please call set_class method first.')

    def ratio(self, key=None):
        """ Dataset ratio: (dimension / number of examples).
        """
        return len(self.columns) / len(self.get_data(key))

    def categorical_ratio(self):
        """ Alias for symbolic_ratio method.
        """
        return self.symbolic_ratio()

    def symbolic_ratio(self):
        """ Ratio of symbolic attributes.
        """
        return len(self.get_data('categorical').columns) / len(self.columns)

    def sparsity(self):
        """ Compute the proportion of zeros.
        """
        return 1 - (np.count_nonzero(self) / self.size)

    def class_deviation(self):
        if self.has_class():
            return self.get_data('y').std().mean()
        else:
            raise Exception('No class is defined. Please use set_class method to define one.')

    def missing_ratio(self):
        """ Ratio of missing values.
        """
        return (self.isnull().sum() / len(self)).mean()

    def descriptors(self):
        """ All in one.
        """
        names = []
        values = []
        if self.has_class():
            names.append('task')
            values.append(self.get_task())
        names.append('shape')
        values.append(self.shape)
        names.append('ratio')
        values.append(self.ratio()) # ratio dimension / number of examples
        names.append('sparsity')
        values.append(self.sparsity()) # proportion of zeros
        if self.has_class():
            names.append('class_deviation')
            values.append(self.class_deviation()) # mean std of target
        names.append('categorical_ratio')
        values.append(self.symbolic_ratio())  # ratio of symbolic attributes
        names.append('missing_ratio')
        values.append(self.missing_ratio())   # ratio of missing values
        names.append('max_skewness')
        values.append(self.skew().max())  # max skewness
        names.append('mean_skewness')
        values.append(self.skew().mean()) # mean skewness
        names.append('min_skewness')
        values.append(self.skew().min()) # min skewness
        return AutoData([values], columns=names)

    ##################################################################


    ## 2. ###################### PROCESSINGS #########################
    # Imputation, encoding, normalization

    # TODO
    # Avoid data leakage during processing
    # train/test/valid split shuffle
    # Dimensionality reduction (method) (only on X)

    def process(self):
        """ Alias for processing method.
        """
        return self.processing()

    def processing(self):
        """ Do basic processings in order to quickly be able to use the dataset.
            It is recommended to use the other processing methods (encoding, normalization)
            to use personalized parameters. This method is kept simple on purpose.
            Steps:
              1. Missing values imputation.
              2. Label encoding for categorical variables.
        """
        data = self.imputation('most')
        data = data.encoding('label', 'categorical')
        return data

    def train_test_split(self,
                         test_size=0.3,
                         shuffle=True,
                         valid=False,
                         valid_size=0.1,
                         train_size=None):
        """ Procedure doing the train/test split and store it into self.indexes.

            :param test_size: proportion of examples in test set.
            :param shuffle: whether to shuffle examples or not.
            :param valid: whether to do a train/valid/test split or not (not implemented yet).
            :param valid_size: proportion of example in validation set (not implemented yet).
            :param train_size: kind of alias which define test_size (1 - train_size).
        """
        if train_size: # TODO
            test_size = 1 - train_size
        N = self.shape[0]
        index = list(range(N))
        if shuffle:
            random.shuffle(index)
        split = round(N * (1 - test_size))
        train_index = index[:split]
        valid_index = []
        test_index = index[split:]
        self.set_indexes('train', train_index)
        self.set_indexes('valid', valid_index)
        self.set_indexes('test', test_index)

    def set_class(self, y=None):
        """ Procedure that defines one or several column(s) representing class / target / y.

            :param y: str or list of str representing column names to use as class(es). If y is None then the target is re-initialized (no class).
        """
        X = list(self)  # column names
        # no class (re-initialize)
        if y is None:
            self.set_indexes('y', [])
        # y is 1 column
        elif isinstance(y, str) or isinstance(y, int):
            self.set_indexes('y', [y])
            try:
                X.remove(y)
            except:
                raise Exception('Column "{}" does not exist.'.format(y))
        # y is array-like
        else:
            self.set_indexes('y', y)
            for name in y:
                try:
                    X.remove(name)
                except:
                    raise Exception('Column "{}" does not exist.'.format(name))
        self.set_indexes('X', X)

    def has_class(self):
        """ Return True if 'y' is defined and corresponds to one column (or more).
        """
        return ('y' in self.indexes.keys()) and (self.indexes['y'] != [])

    def has_split(self):
        """ Return True if 'train' and 'test' are defined (or more).
        """
        train = ('train' in self.indexes.keys()) and (self.indexes['train'] != [])
        test = ('test' in self.indexes.keys()) and (self.indexes['test'] != [])
        return (train and test)

    def impute(self, **kwargs):
        """ Alias for imputation method.
        """
        return self.imputation(**kwargs)

    def imputation(self, method='most', key=None):
        """ Impute missing values.

            :param method: None, 'remove', 'most', 'mean', 'median'
            :return: Data with imputed values.
            :rtype: AutoData
        """
        data = self.copy()
        rows, columns = self.get_index(key) # get_index instead of get_data to modify wanted columns and keep others
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

    def normalize(self, **kwargs):
        """ Alias for normalization method.
        """
        return self.normalization(**kwargs)

    def normalization(self, method='standard', key='numerical', split=True, **kwargs):
        """ Normalize data.

            :param method: 'standard', 'min-max', None, 'binarize'
            :param key: Subset of data to encode. Put to None for all columns. 'numerical' by default.
            :param split: If False, do the processing on the whole frame without train/test split.
            :return: Normalized data.
            :rtype: AutoData
        """
        data = self.copy()
        has_split = self.has_split() and split
        rows, columns = self.get_index(key)
        # avoid data leakage (apply processing on train and then on test with same parameters)
        if has_split:
            train = data.get_data('train')
            test = data.get_data('test')
        # process
        if method in ['binary', 'binarize', 'binarizer']: # not column wise
            return data.transform(Binarizer, key=key, **kwargs)
        for column in columns:
            if method in ['standard', 'std']:
                if has_split:
                    train, (mean, std) = normalization.standard(train, column, return_param=True)
                    test = normalization.standard(test, column, mean=mean, std=std)
                else:
                    data = normalization.standard(data, column)
            elif method in ['min-max', 'minmax', 'min_max']:
                if has_split:
                    train, (mini, maxi) = normalization.min_max(train, column, return_param=True)
                    test = normalization.min_max(test, column, mini=mini, maxi=maxi)
                else:
                    data = normalization.min_max(data, column)
            else:
                raise Exception('Unknown normalization method: {}'.format(method))
        if has_split:
            data = from_train_test(train, test)
        #data.flush_index()
        return data

    def encode(self, **kwargs):
        """ Alias for encoding method.
        """
        return self.encoding(**kwargs)

    def encoding(self, method='label', key='categorical', target=None, split=True, **kwargs):
        """ Encode (categorical) variables.

            :param method: 'none', 'label', 'one-hot', 'rare-one-hot', 'target', 'likelihood', 'count', 'probability', 'binarize'
            :param key: Subset of data to encode. Put to None for all columns. 'categorical' by default.
            :param target: Target column name (target encoding).
            :param coeff: Coefficient defining rare values (rare one-hot encoding).
                          A rare category occurs less than the (average number of occurrence * coefficient).
            :param split: If False, do the processing on the whole frame without train/test split.
        """
        data = self.copy()
        has_split = self.has_split() and split and (method in ['likelihood', 'count', 'target'])
        rows, columns = self.get_index(key)
        # avoid data leakage (apply processing on train and then on test with same parameters)
        if has_split: # can be more memory efficient
            train = data.get_data('train')
            test = data.get_data('test')
        # process
        if method in ['binary', 'binarize', 'binarizer']: # not column wise
            return data.transform(Binarizer, key=key, **kwargs)
        for column in columns:
            if method in ['none', 'drop'] or method is None:
                data = encoding.none(data, column)
            elif method == 'label':
                data = encoding.label(data, column)
            elif method in ['onehot', 'one_hot', 'one-hot']:
                data = encoding.one_hot(data, column) # TODO: fix class behaviour
            elif method == 'likelihood':
                if has_split:
                    train, mapping = encoding.likelihood(train, column, return_param=True)
                    test = encoding.likelihood(test, column, mapping=mapping)
                else:
                    data = encoding.likelihood(data, column)
            elif method == 'count':
                if has_split:
                    train, mapping = encoding.count(train, column, return_param=True)
                    test = encoding.count(test, column, mapping=mapping)
                else:
                    data = encoding.count(data, column)
            elif method == 'target':
                if target is None:
                    if not self.has_class():
                        raise Exception('You need to specify a target column or to set a class to use target encoding.')
                    else:
                        target = self.indexes['y'][0]
                        num_classes = len(self.indexes['y']) # TODO: multiclass?
                        if num_classes > 1:
                            print('WARNING: only 1 over {} classes will be used for the target encoding.'.format(num_classes))
                if has_split:
                    train, mapping = encoding.target(train, column, target, return_param=True)
                    test = encoding.target(test, column, target, mapping=mapping)
                else:
                    data = encoding.target(data, column, target)
            else:
                raise Exception('Unknow encoding method: {}'.format(method))
        if has_split:
            data = from_train_test(train, test)
        if encoding not in ['label', 'count']:
            data.flush_index() # update columns indexes for encoding that change number of columns
        return data

    def pca(self, key=None, return_param=False, model=None, verbose=False, **kwargs):
        """ Compute Principal Components Analysis.
            Use kwargs for additional PCA parameters (cf. sklearn doc).
            /!\ WARNING: redundancy with 'transform' method.

            :param key: Indexes key to select data.
            :param return_param: If True, returns a tuple (X, pca) to store PCA parameters and apply them later.
            :param model: Use this argument to pass a trained PCA model.
            :param verbose: Display additional information during run.
            :rtype: AutoData
            :return: Transformed data
        """
        data = self.copy()
        indexes = data.indexes
        # compute PCA and copy indexes
        if return_param:
            data, pca = reduction.pca(data, key=key, return_param=return_param, model=model, verbose=verbose, **kwargs)
        else:
            data = reduction.pca(data, key=key, return_param=return_param, model=model, verbose=verbose, **kwargs)
        data = AutoData(data, indexes=indexes)
        # variable are now only numerical
        data.indexes['categorical'] = []
        data.indexes['numerical'] = list(data)
        data.flush_index() # update columns index after dimensionality change
        if return_param:
            return data, pca
        return data

    def tsne(self, key=None, verbose=False, **kwargs):
        """ Compute T-SNE.

            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for T-SNE (see sklearn doc)
            :return: Transformed data
            :rtype: AutoData
        """
        data = AutoData(reduction.tsne(self, key=key, verbose=verbose, **kwargs))
        data.flush_index()
        return data

    def lda(self, key=None, verbose=False, **kwargs):
        """ Compute Linear Discriminant Analysis.

            :param verbose: Display additional information during run
            :param **kwargs: Additional parameters for LDA (see sklearn doc)
            :return: Transformed data
            :rtype: AutoData
        """
        if 'y' not in self.indexes:
            raise Exception('No class is defined. Please use set_class method to define one.')
        data = AutoData(reduction.lda(self, key=key, verbose=verbose, **kwargs))
        data.flush_index()
        return data

    def reduction(self, method='pca', key=None, verbose=False, **kwargs):
        """ Dimensionality reduction.

            :param method: 'pca', 'lda', 'tsne', 'hashing', 'factor', 'random'
        """
        data = self.get_data(key)
        if method == 'pca':
            data = data.pca(key=key, verbose=verbose, **kwargs)
        elif method == 'tsne':
            data = data.tsne(key=key, verbose=verbose, **kwargs)
        elif method == 'lda':
            data = data.lda(key=key, verbose=verbose, **kwargs)
        elif method in ['hashing', 'feature_hashing', 'feature-hashing']:
            data = AutoData(reduction.feature_hashing(data, key=key, **kwargs))
        elif method in ['factor', 'factor-analysis', 'factor_analysis']:
            data = data.transform(FactorAnalysis, key=key, **kwargs)
        elif method in ['random_projection', 'random', 'gaussian_random_projection']:
            data = data.transform(GaussianRandomProjection, key=key, **kwargs)
        else:
            raise Exception('Unknown dimensionality reduction method: {}'.format(method))
        return data

    def transform(self, model, key=None, return_param=False, **kwargs):
        """ General method to apply a transformation on data.
            It can be normalization, dimensionality reduction, etc.
            You can use kwargs for additional parameters of the transformer.

            :param model: The transformer. It must have 'fit' and 'transform' methods.
                          If it is an object instead of a class, model is considered to be an already fitted transformer.
            :param return_param: If true, returns the fitted transformer.
        """
        data = self.copy()
        indexes = data.indexes
        columns = data.columns
        try: # initialize object
            transformer = model(**kwargs)
            transformer.fit(data)
        except TypeError: # object not callable, probably already declared
            transformer = model
        data = transformer.transform(data)
        try:
            data = AutoData(data, indexes=indexes, columns=columns)
        except: # not the same number of columns
            data = AutoData(data, indexes=indexes)
            data.flush_index() # update columns index after dimensionality change
        data.get_types() # need to compute the types
        if return_param:
            return data, transformer
        else:
            return data

    ## 3. ###################### VISUALIZATION #######################

    # TODO
    # Class coloration on plots!

    def plot(self, key=None, ad=None, c=None, save=None, **kwargs):
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
        visualization.plot(self, key=key,
                                 ad=ad, c=c,
                                 save=save, **kwargs)

    def pairplot(self, key=None, max_features=12, force=False, save=None, **kwargs):
        """ Plot pairwise relationships between features.

            :param key: Key for subset selection (e.g. 'X_train' or 'categorical')
            :param max_features: Max number of features to pairplot.
            :param force: If True, plot the graphs even if the number of features is grater than max_features.
            :param save: Path/filename to save figure (if not None)
        """
        visualization.pairplot(self, key=key,
                                     max_features=max_features,
                                     force=force, save=save, **kwargs)

    def plot_pca(self, key):
        self.pca(key, n_components=2).plot()

    def heatmap(self, **kwargs):
        visualization.heatmap(self, **kwargs)

    def correlation(self, **kwargs):
        visualization.correlation(self, **kwargs)

    def compare_marginals(self, ad, **kwargs):
        visualization.compare_marginals(self, ad, **kwargs)

    def hierarchical_clustering(self, **kwargs):
        visualization.hierarchical_clustering(self, **kwargs)

    ## 4. ######################## BENCHMARK ##########################

    # TODO
    # Score reports, confusion matrix

    def score(self, model=None, metric=None, method='baseline', test=None, fit=True, verbose=False):
        """ Benchmark, a.k.a. Utility. This method returns the score of a baseline on the dataset.

            Return the metric score of a model trained and tested on data.
            If a test set is defined ('test' parameter), the model is trained on 'data' and tested on 'test'.

            :param model: Model to fit and test on data.
            :param metric: scoring function.
            :param method: 'baseline' or 'auto'. Useful only if model is None.
            :param fit: If True, fit the model.
            :param test: Test is an optional DataFrame to use as the test set.
            :param verbose: If True, prints model information, classification report and metric function.
        """
        return benchmark.score(self, model=model, metric=metric, method=method, test=test, fit=fit, verbose=verbose)

    # Distribution comparator
    def distance(self, data, method=None, **kwargs):
        """ Distance between two AutoData frames.
            TODO: There are methods to add (cf. utilities.py and metric.py)
            Usage example: ad1.distance(ad2, method='privacy')

            :param data: Second distribution to compare with
            :param method: 'none' (nn_discrepancy), 'discriminant'
        """
        if (method is None) or method in ['None', 'none']:
            return metric.nn_discrepancy(self, data)
        elif method in ['adversarial_accuracy', 'nnaa']:
            return metric.nnaa(self, data, **kwargs)
        elif method in ['discriminant', 'discrepancy', 'adversarial_score']:
            return metric.discriminant(self, data, **kwargs)
        else:
            raise Exception('Unknown distance metric: {}'.format(method))

    def generate(self, method=None):
        """ Fit a generator and generate data with default/kwargs parameters.
            TODO

            :param method: ANM, GAN, VAE, Copula, etc.
        """
        pass

    #################### ALIAS #######################
    # How to automate this ?
    # __getattr__ ?
    # Need to think about this
    #X(), y(), X_train, y_test, numerical, header, etc.
    # set_X ?
    ##################################################
