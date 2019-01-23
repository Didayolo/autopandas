import numpy as np
import pandas as pd
from encoding import *
from normalization import *

def get_types(df):
    """ Get variables types: Numeric, Binary or Categorical.
    
        :param df: pandas DataFrame
        :return: List of type of each variable
        :rtype: list
    """
    
    x = df.copy()
    dtypes = list()
    
    n_rows = df.shape[0]
    prop = int(n_rows / 20) # Arbitrary proportion of different values where a numerical variable is considered categorical
    
    for column in x.columns:
        n = len(x[column].unique())
        #try:
        #    s = x[column].sum()
        #except:
        #    s = -np.inf
        if n == 2:
            dtypes.append('Binary')
        #elif (n > 2 and (sum == n*(n-1)/2 or sum == n*(n+1)/2)) or any(isinstance(i, str) for i in x[column]): ???
        elif (n > 2 and (n <= prop)) or any(isinstance(i, str) for i in x[column]):
            dtypes.append('Categorical')
        else:
            dtypes.append('Numerical')
    return dtypes
