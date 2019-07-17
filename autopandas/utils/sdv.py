# Synthetic Data Vault

# Processing (encode)
# Back-processing (decode) + clipping?

from scipy.stats import truncnorm
import autopandas

def encode(data):
    """ Encode the data into SDV format.
    """
    # loop through every column
    limits = {}
    min_max = {}
    data = data.copy()
    for c in data.columns:
        if c in data.indexes['categorical']: # if categorical
            data[c], lim = categorical(data[c])
            limits[c] = lim
        else: # if numerical
            data[c], min_res, max_res = numeric(data[c])
            min_max[c] = (min_res, max_res)
    return data, limits, min_max

def decode(new_data, data, limits, min_max):
    """ Decode the data from SDV format.

        :param data: Data in SDV format
        :param data: Original data
        :param limits: Limits returned by sdv.encode
        :param min_max: Min-max returned by sdv.encode
    """
    new_data = autopandas.AutoData(new_data, columns=data.columns, indexes=data.indexes)
    for c in new_data.columns:
        if c in limits:
            new_data[c] = undo_categorical(new_data[c], limits[c])
        else:
            new_data[c] = undo_numeric(new_data[c], *min_max[c])
    return new_data

def categorical(column):
    """ Convert a categorical column to continuous.
    """
    # get categories
    categories = (column.value_counts() / len(column)).sort_values(ascending=False)
    # get distributions to pull from
    distributions = {}
    limits = {}
    a = 0
    # for each category
    for cat, val in categories.iteritems():
        # figure out the cutoff value
        b = a + val
        # create the distribution to sample from
        mu, sigma = (a + b) / 2, (b - a) / 6
        distributions[cat] = truncnorm((a - mu) / sigma,
                                       (b - mu) / sigma,
                                       mu, sigma)
        limits[b] = cat
        a = b
    # sample from the distributions and return that value
    return column.apply(lambda x: distributions[x].rvs()), limits

def numeric(column):
    """ Normalize a numerical column.
    """
    return ((column - min(column)) / (max(column) - min(column))), min(column), max(column)

def undo_categorical(column, lim):
    """ Convert a categorical column to continuous.
    """
    def cat_decode(x, limits):
        """ Decoder for categorical data.
        """
        for k, v in limits.items():
            if x < k:
                return v
    return column.apply(lambda x: cat_decode(x, lim))

def undo_numeric(column, min_column, max_column):
    """ Normalize a numerical column.
    """
    return ((max_column - min_column) * column) + min_column
