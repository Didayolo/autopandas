# Functions for missing data imputation

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
        :param column: Column to impute
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
