# Benchmark Functions

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

def score(data, model=None, metric=None, method='baseline', fit=True):
    """ Benchmark ...
    """
    if model is None:

        # Select model
        if method == 'baseline':
            clf = RandomForestClassifier()
            reg = RandomForestRegressor()

        elif method in ['autosklearn', 'automl', 'automatic']:
            clf = AutoSklearnClassifier()
            reg = AutoSklearnRegressor() # multi-ouput ??

        else:
            raise Exception('Unknown method: {}'.format(method))

        # Select task
        task = data.get_task()
        if  task == 'classification':
            model = clf

        elif task == 'regression':
            model = reg

        else:
            raise Exception('Unknown task: {}.'.format(task))

    if 'test' not in data.indexes:
        raise Exception('No train/test split. Please use train_test_split method before calling score.')

    elif 'y' not in data.indexes:
        raise Exception('No class. Please use set_class method before calling score.')

    # Let's go!
    else:
        X_train = data.get_data('X_train')
        y_train = data.get_data('y_train')
        X_test = data.get_data('X_test')
        y_test = data.get_data('y_test')

        # mono-class
        if y_train.shape[1] == 1:
            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()

        if fit:
            model.fit(X_train, y_train)

        # todo: different scoring metrics
        return model.score(X_test, y_test)
