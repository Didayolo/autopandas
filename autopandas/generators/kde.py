# Parzen Windows Kernel Density Estimation

# Interface to scikit-learn implementation
from sklearn.neighbors import KernelDensity
import autopandas

class KDE():
    def __init__(self, **kwargs):
        """ Kernel Density Estimation (parzen windows).
        """
        self.model = KernelDensity(**kwargs)
        self.columns = None
        self.indexes = None

    def fit(self, data, **kwargs):
        """ Train the generator with data.

            :param data: The training data.
        """
        self.columns = data.columns
        self.indexes = data.indexes
        self.model.fit(data, **kwargs)

    def sample(self, n=1, **kwargs):
        """ Sample from trained KDE.

            :param n: Number of examples to sample.
        """
        if self.indexes is None:
            raise Exception('You firstly need to train the KDE before sampling. Please use fit method.')
        else:
            gen_data = self.model.sample(n, **kwargs)
            return autopandas.AutoData(gen_data, columns=self.columns, indexes=self.indexes)
