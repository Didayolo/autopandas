# Benchmark Functions

import numpy as np
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from autosklearn.classification import AutoSklearnClassifier
#from autosklearn.regression import AutoSklearnRegressor

# TODO:
# - Train error bars
# - Test error bars
# - Score can take list of metrics, list of models

def score_error_bars(data, n=10, model=None, metric=None, method='baseline', fit=True, test=None, verbose=False):
    """ Run score method several times to compute error bars.
        The parameters are the same as score method.
        TODO: optimize computation.
        TODO: cross-val
        :return: mean and variance.
    """
    scores = []
    for _ in range(n):
        scores.append(score(data, model=model, metric=metric, method=method, fit=fit, test=test, verbose=verbose))
    mean = np.mean(scores)
    var = np.var(scores)
    return mean, var

def score(data, model=None, metric=None, method='baseline', fit=True, test=None, average='weighted', verbose=False):
    """ Benchmark, a.k.a. Utility.

        Return the metric score of a model trained and tested on data.
        If a test set is defined ('test' parameter), the model is trained on 'data' and tested on 'test'.

        :param model: Model to fit and test on data.
        :param metric: scoring function.
        :param method: 'baseline' or 'auto'. Useful only if model is None.
        :param fit: If True, fit the model.
        :param test: Test is an optional DataFrame to use as the test set.
        :param average: Method for averaging the multi-class One-vs-One metrics scheme.
        :param verbose: If True, prints model information, classification report and metric function.

        :rtype: float
        :return: Metric score of the model trained and tested on data.
    """
    if 'y' not in data.indexes:
        raise Exception('No class defined. Please use set_class method before calling score.')
    if model is None:
        # Select model
        if method == 'baseline':
            clf = RandomForestClassifier()
            reg = RandomForestRegressor()
        elif method in ['auto', 'autosklearn', 'automl', 'automatic']:
            raise Exception('autosklearn got removed from requirements. This method is currently not implemented.')
            #clf = AutoSklearnClassifier()
            #reg = AutoSklearnRegressor() # multi-ouput ??
        else:
            raise Exception('Unknown method: {}'.format(method))
        # Select task
        task = data.get_task()
        if  task == 'classification':
            model = clf
            if metric is None:
                metric = accuracy_score
        elif task == 'regression':
            model = reg
            if metric is None:
                metric = r2_score
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
    ### /!\ TODO: CLEAN CODE BELOW /!\ ###
    try:
        y_pred = model.predict_proba(X_test) # SOFT
        try:
            score = metric(y_test, y_pred, average=average, multi_class='ovo') #labels=np.unique(y_pred))
        except:
            try:
                score = metric(y_test, y_pred, average=average)
            except:
                score = metric(y_test, y_pred)
    except:
        y_pred = model.predict(X_test) # HARD
        try:
            score = metric(y_test, y_pred, average=average, multi_class='ovo')
        except:
            try:
                score = metric(y_test, y_pred, average=average)
            except:
                try:
                    score = metric(y_test, y_pred)
                except:
                    labels = np.unique(y_pred)
                    score = metric(y_test, y_pred, labels=labels)
    if verbose:
        print(model)
        print(classification_report(y_test, y_pred))
        print('Metric: {}'.format(metric))
    return score
