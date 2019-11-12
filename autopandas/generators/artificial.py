# Artificial data generator

#from ..autopandas import from_X_y
import autopandas
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_circles

class Artificial():
    def __init__(self, method='moons'):
        """ Artificial data generator.
            Generate 2D classification datasets.

            :param method: 'moons', 'blobs' or 'circles'.

        """
        self.method = method

    def sample(self, n=1, noise=0.01):
        """ Sample data from the artificial data generator.

            :param n: Number of artificial points to create.
        """
        if self.method == 'moons':
            X, y = make_moons(n_samples=n, noise=noise)
        elif self.method == 'blobs':
            X, y = make_blobs(n_samples=n, centers=3, n_features=2)
        elif self.method == 'circles':
            X, y = make_circles(n_samples=n, noise=noise)
        else:
            raise Exception('Unknown method: {}'.format(method))
        # Create AutoData frame from X and y
        data = autopandas.from_X_y(X, y)
        return data
