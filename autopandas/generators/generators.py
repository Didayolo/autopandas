# Generators module

# Import all generators
from .artificial import Artificial
from .copycat import Copycat
from .vae import VAE
from .anm import ANM
from .copula import Copula

# Gaussian Mixture and Parzen Widows from scikit-learn
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KernelDensity as KDE

# TODO
# GAN, WGAN, medGAN
# SAM
