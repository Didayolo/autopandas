import pandas as pd

def mean(df, column):
    """ Replace missing values by the mean of the column
    
        :param df: Data as a pd.DataFrame
        :param column: Column to impute
        :return: Imputed data
        :rtype: pd.DataFrame
    """
    x = df.copy()
    x[column] = x[column].fillna(x[column].mean())
    return x

def median(df, column):
    """ Replace missing values by the median of the column
    
        :param df: Data as a pd.DataFrame
        :param column: Column to impute
        :return: Imputed data
        :rtype: pd.DataFrame
    """
    x = df.copy()
    x[column] = x[column].fillna(x[column].median())
    return x

def remove(df, columns):
    """ Remove missing values
    
        :param df: Data as a pd.DataFrame
        :param column: Column to impute
        :return: Imputed data
        :rtype: pd.DataFrame
    """
    x = df.copy()
    x = x.dropna(axis=0, subset=columns)
    return x

def most(df, column):
    """ Replace missing values by the most frequent value of the column
    
        :param df: Data as a pd.DataFrame
        :param column: Column to impute
        :return: Imputed data
        :rtype: pd.DataFrame
    """
    x = df.copy()
    most_frequent_value = x[column].value_counts().idxmax()
    x[column] = x[column].fillna(most_frequent_value)
    return x
