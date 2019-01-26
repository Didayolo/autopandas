# Benchmark Functions

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

def score(data, model=None, metric=None):
    """ Benchmark ...
    """
    if model is None:
        if data.get_task() == 'classification':
            model = RandomForestClassifier()

        elif data.get_task() == 'regression':
            model = RandomForestRegressor()

        else:
            raise Exception('Unknown task.')

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

        model.fit(X_train, y_train)
        return model.score(X_test, y_test)
