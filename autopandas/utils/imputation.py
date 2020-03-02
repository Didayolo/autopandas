# Functions for missing data imputation

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def mean(data, column):
    """ Replace missing values by the mean of the column.

        :param data: AutoData data
        :param column: Column to impute
        :return: Imputed data
        :rtype: autopandas.AutoData
    """
    data[column] = data[column].fillna(data[column].mean())
    return data

def median(data, column):
    """ Replace missing values by the median of the column.

        :param data: AutoData data
        :param column: Column to impute
        :return: Imputed data
        :rtype: autopandas.AutoData
    """
    data[column] = data[column].fillna(data[column].median())
    return data

def remove(data, columns):
    """ Simply remove columns containing missing values.

        :param data: AutoData data
        :param columns: Column(s) to impute
        :return: Imputed data
        :rtype: autopandas.AutoData
    """
    data = data.dropna(axis=0, subset=columns)
    return data

def most(data, column):
    """ Replace missing values by the most frequent value of the column.

        :param data: AutoData data
        :param column: Column to impute
        :return: Imputed data
        :rtype: autopandas.AutoData
    """
    most_frequent_value = data[column].value_counts().idxmax()
    data[column] = data[column].fillna(most_frequent_value)
    return data

def infer(data, column, model=None, return_param=False, fit=True):
    """ Replace missing values by the most frequent value of the column.

        :param data: AutoData data
        :param column: Column to impute
        :param model: Predictive model to train and use for imputation
        :param return_param: If True, returns a tuple (data, model) to store fitted model and apply it later
        :param fit: If True, fit the model
        :return: Imputed data
        :rtype: autopandas.AutoData
    """
    if model is None:
        if column in data.indexes['numerical']: # works only for AutoData
            model = RandomForestRegressor()
        else:
            model = RandomForestClassifier()
    nan_rows = np.isnan(data[column]) # rows to impute
    if np.any(nan_rows): # if there are NaNs to impute
        if fit:
            train = data[np.invert(nan_rows)].dropna(axis=0) # rows without NaN, TODO
            y_train, X_train = train[column], train.drop(column, axis=1)
            model.fit(X_train, y_train)
        test = data[nan_rows].fillna(0) # TODO
        y_test, X_test = test[column], test.drop(column, axis=1)
        y_pred = model.predict(X_test)
        data.loc[nan_rows, column] = y_pred
    if return_param:
        return data, model
    return data
