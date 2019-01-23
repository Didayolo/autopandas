import pandas as pd

def standard(df, column, mean=None, std=None, return_param=False):
    """
        Performs standard normalization.
            
        :param df: Data
        :param column: Column to normalize
        :param mean: Mean, computed if not specified
        :param std: Standard deviation, computed if not specified
        :param return_param: if True, mean and std are returned
        :return: Normalized data
        :rtype: pd.DataFrame
    """
    x = df.copy()
    if not mean and not std:
        mean = x[column].mean()
        std = x[column].std()
    x[column] = (x[column] - mean) / std
    
    if return_param:
        return x, (mean, std)
    return x

def min_max(df, column, mini=None, maxi=None, return_param=False):
    """
        Performs min-max normalization.
            
        :param df: Data
        :param column: Column to normalize
        :param mini: Minimum, computed if not specified
        :param maxi: Maximum, computed if not specified
        :param return_param: if True, mean and std are returned
        :return: Normalized data
        :rtype: pd.DataFrame
    """
    x = df.copy()
    if not mini and not maxi:
        mini = x[column].min()
        maxi = x[column].max()
    x[column] = (x[column] - mini) / (maxi - mini)
    
    if return_param:
        return x, (mini, maxi)
    return x
