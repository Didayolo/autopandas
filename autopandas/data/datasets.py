# Module for loading toy datasets

import autopandas as apd
import os
# TODO: add documentation, author names...

CURRENT_PATH = os.path.dirname(__file__)

def load_iris():
    """ Iris dataset.
        https://archive.ics.uci.edu/ml/datasets/Iris
    """
    data = apd.read_csv(os.path.join(CURRENT_PATH, 'iris.csv'))
    data = data.encoding()
    data.set_class('Species')
    return data

def load_wine():
    """ Wine quality dataset.
        https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    """
    data = apd.read_csv(os.path.join(CURRENT_PATH, 'wine.csv'))
    data.set_class('quality')
    return data

def load_adult():
    """ Adult Income dataset.
        https://archive.ics.uci.edu/ml/datasets/Adult
    """
    data = apd.read_csv(os.path.join(CURRENT_PATH, 'adult.csv'))
    data = data.encoding()
    data.set_class('income')
    return data

def load_mushrooms():
    """ Mushrooms dataset.
        https://archive.ics.uci.edu/ml/datasets/mushroom
    """
    data = apd.read_csv(os.path.join(CURRENT_PATH, 'mushrooms.csv'))
    data = data.encoding()
    data.set_class('class')
    return data

def load_diabetes():
    """ Pima Indians Diabetes Dataset.
        https://www.kaggle.com/uciml/pima-indians-diabetes-database
    """
    data = apd.read_csv(os.path.join(CURRENT_PATH, 'diabetes.csv'), header=None)
    data.set_class(8)
    return data

def load_seeds():
    """ Seeds dataset.
        https://archive.ics.uci.edu/ml/datasets/seeds
    """
    data = apd.read_csv(os.path.join(CURRENT_PATH, 'seeds.csv'))
    data.set_class('V8')
    return data

def load_titanic():
    """ Titanic dataset.
        https://www.kaggle.com/c/titanic
    """
    print('WARNING: target is missing for test set.')
    train = apd.read_csv(os.path.join(CURRENT_PATH, 'titanic_train.csv'))
    test = apd.read_csv(os.path.join(CURRENT_PATH, 'titanic_test.csv'))
    data = apd.from_train_test(train, test)
    data = data.encoding()
    data = data.imputation()
    data.set_class('Survived')
    return data

def load_boston():
    """ Boston housing dataset.
        https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
    """
    print('WARNING: target is missing for test set.')
    train = apd.read_csv(os.path.join(CURRENT_PATH, 'boston_train.csv'))
    test = apd.read_csv(os.path.join(CURRENT_PATH, 'boston_test.csv'))
    data = apd.from_train_test(train, test)
    data = data.imputation()
    data.set_class('medv')
    return data

def load_squares(n=1):
    """ Squares dataset by Adrien PAVAO.
    """
    if n==1:
        data = apd.read_csv(os.path.join(CURRENT_PATH, 'squares1.csv'), header=None)
    elif n==2:
        data = apd.read_csv(os.path.join(CURRENT_PATH, 'squares2.csv'), header=None)
    else:
        raise Exception('n argument only accepts values 1 and 2')
    return data
