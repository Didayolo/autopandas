# Example generator. Copycat simply copy training data.

class Copycat():
    def __init__(self):
        """ Baseline generator: simply copy training data.
        """
        self.data = None

    def fit(self, data):
        """ Train the generator with data.

            :param data: The data to copy.
        """
        self.data = data

    def sample(self, n=1, replace=False):
        """ Sample from train data.

            :param n: Number of examples to sample.
            :param replace: If True, sample with replacement.
        """
        if self.data is None:
            raise Exception('You firstly need to train the Copycat before sampling. Please use fit method.')
        else:
            return self.data.sample(n, replace=replace)
