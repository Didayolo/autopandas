# AutoData
# Based on Pandas, numpy, sklearn, matplotlib

#Lecture AutoML, CSV, TFRecords

# Simplification (une méthode par affichage, traitement, etc)
# Le notebook appelle chaque méthode avec une explication
# Transformations : méthodes qui renvoie un objet AutoML (pas de duplication systématique des données)

# Init info, etc.
# Train, test, split

# Classification or regression ?

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def read_csv(*args, **kwargs):
    return AutoData(pd.read_csv(*args, **kwargs))

def read_automl():
    pass

# Surcouche de pd.DataFrame
class AutoData(pd.DataFrame):
    """ AutoData class
    """

    ## 1. #################### READ/WRITE DATA ######################
    def __init__(self, *args, **kwargs):
        """ Constructor
        """
        pd.DataFrame.__init__(self, *args, **kwargs)

        self.info = {}
        self.indexes = {'header':range(5)} # header, train, test, valid, X, y

    def to_automl(self):
        pass


    def set_index(self, key, value):
        self.indexes[key] = value


    def get_data(self, key=None):
        """ Get data
        """
        if key is None:
            return self

        elif key in ['train', 'valid', 'test', 'header']:
            return self.iloc[self.indexes[key]]

        elif key in ['X', 'y']:
            return self.loc[:, self.indexes[key]]

        elif '_' in key:
            # X_train, y_test, etc.
            v, h = key.split('_')
            hindex = self.indexes[h]
            vindex = self.indexes[v]
            return self.loc[hindex, vindex]

        else:
            raise Exception('Unknown key.')



    ## 2. ###################### PROCESSINGS #########################
    # Imputation, encoding, normalization

    def train_test_split(self, test_size=0.3, shuffle=True):
        """ Procedure
        """
        N = self.shape[0]
        split = round(N * (1 - test_size))
        self.set_index('train', range(split))
        self.set_index('test', range(split, N))


    def set_class(self, y):
        """ Procedure
        """
        self.set_index('y', y)
        X = list(self) # column names

        if isinstance(y, str) or isinstance(y, int):
            X.remove(y)

        else:
            for name in y:
                X.remove(name)

        self.set_index('X', X)


    def process(imputation=None, normalization=None, encoding=None):
        pass


    ## 3. ###################### VISUALIZATION #######################
    def pca():
        pass


    def pairplot():
        pass


    ## 4. ######################## BENCHMARK ##########################
    def score(self, clf=RandomForestClassifier, metric=None):
        if 'test' not in self.esets:
            raise Exception('No train/test split.')

        elif 'y' not in self.fsets:
            raise Exception('No class.')

        # Let's go!
        else:
            X = self.get_data('X')
            y = self.get_data('y')
            clf.fit(X, y)
