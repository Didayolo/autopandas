# Example generator. Copycat simply copy training data.

class Copycat():
    """ Baseline generator: simply copy training data
    """
    def __init__(self):
        self.data = None

    def fit(self, data):
        self.data = data

    def sample(self, n=1, replace=False):
        if self.data is None:
            raise Exception('You firstly need to train the Copycat before sampling. Please use fit method.')
        else:
            return self.data.sample(n, replace=replace)
