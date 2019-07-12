# Benchmark Functions

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor

def score(data, model=None, metric=None, method='baseline', fit=True, test=None, verbose=False):
    """ Benchmark, a.k.a. Utility.

        Return the metric score of a model trained and tested on data.
        If a test set is defined ('test' parameter), the model is trained on 'data' and tested on 'test'.

        Parameters
        ----------
        model : Model to fit and test on data.
        metric : scoring function.
        method : 'baseline' or 'auto'. Useful only if model is None.
        fit : If True, fit the model.
        test : Test is an optional DataFrame to use as the test set.
        verbose : If True, prints model information, classification report and metric function.

        Returns
        -------
        float
            Metric score of the model trained and tested on data.
    """
    if 'y' not in data.indexes:
        raise Exception('No class defined. Please use set_class method before calling score.')
    if metric is None:
        metric = accuracy_score
    if model is None:
        # Select model
        if method == 'baseline':
            clf = RandomForestClassifier()
            reg = RandomForestRegressor()
        elif method in ['auto', 'autosklearn', 'automl', 'automatic']:
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
    if test is not None: # test set is defined
        X_train = data.get_data('X')
        y_train = data.get_data('y')
        X_test = test.get_data('X')
        y_test = test.get_data('y')
    else:
        if 'test' not in data.indexes:
            raise Exception('No train/test split. Please use train_test_split method before calling score.')
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
    # Let's go!
    y_pred = model.predict(X_test)
    if verbose:
        print(model)
        print(classification_report(y_test, y_pred))
        print('Metric: {}'.format(metric))
    try:
        score = metric(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    except:
        score = metric(y_test, y_pred)
    return score
    #return model.score(X_test, y_test)
