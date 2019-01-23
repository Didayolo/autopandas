# Functions for missing data imputation

def mean(data, column):
    """ Replace missing values by the mean of the column
        :param data: AutoData data
        :param column: Column to impute
        :return: Imputed data
        :rtype: AutoData
    """
    x = data.copy()
    x[column] = x[column].fillna(x[column].mean())
    return x

def median(data, column):
    """ Replace missing values by the median of the column
        :param data: AutoData data
        :param column: Column to impute
        :return: Imputed data
        :rtype: AutoData
    """
    x = data.copy()
    x[column] = x[column].fillna(x[column].median())
    return x

def remove(data, columns):
    """ Remove missing values
        :param data: AutoData data
        :param column: Column to impute
        :return: Imputed data
        :rtype: AutoData
    """
    x = data.copy()
    x = x.dropna(axis=0, subset=columns)
    return x

def most(data, column):
    """ Replace missing values by the most frequent value of the column
        :param data: AutoData data
        :param column: Column to impute
        :return: Imputed data
        :rtype: AutoData
    """
    x = data.copy()
    most_frequent_value = x[column].value_counts().idxmax()
    x[column] = x[column].fillna(most_frequent_value)
    return x
