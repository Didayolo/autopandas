# Function for normalization

def standard(data, column, mean=None, std=None, return_param=False):
    """ Performs standard normalization.

        :param data: Data
        :param column: Column to normalize
        :param mean: Mean, computed if not specified
        :param std: Standard deviation, computed if not specified
        :param return_param: if True, mean and std are returned
        :return: Normalized data
        :rtype: autopandas.AutoData
    """
    if not mean and not std:
        mean = data[column].mean()
        std = data[column].std()
    data[column] = (data[column] - mean) / std
    if return_param:
        return data, (mean, std)
    return data

def min_max(data, column, mini=None, maxi=None, return_param=False):
    """ Performs min-max normalization.

        :param data: Data
        :param column: Column to normalize
        :param mini: Minimum, computed if not specified
        :param maxi: Maximum, computed if not specified
        :param return_param: if True, mean and std are returned
        :return: Normalized data
        :rtype: autopandas.AutoData
    """
    if not mini and not maxi:
        mini = data[column].min()
        maxi = data[column].max()
    data[column] = (data[column] - mini) / (maxi - mini)
    if return_param:
        return data, (mini, maxi)
    return data
