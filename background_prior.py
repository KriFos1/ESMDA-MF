"""Original stationary Gaussian prior for an independent 2D inversion."""

import numpy as np
from scipy.linalg import solve_triangular

from lm_laplace import SpatialPrior
from rolling_prior import axis_covariance


class BackgroundPrior(SpatialPrior):
    def __init__(self, shape, spacing, lengths, means, stds):
        cells = int(np.prod(shape))
        self.shape = (2, *shape)
        self.mean = np.repeat(means, cells)
        self.std = np.repeat(stds, cells)
        self.factors = []
        for block, kernel in zip(lengths, ('squared-exponential', 'exponential')):
            self.factors.append([
                np.linalg.cholesky(axis_covariance(np.arange(size), step, length, kernel))
                for size, step, length in zip(shape, spacing, block)])

    def quadratic(self, offset):
        """Evaluate the smooth-prior penalty as a positive whitened squared norm."""
        fields = (offset / self.std).reshape(self.shape)
        for block, factors in enumerate(self.factors):
            fields[block] = self._apply_block(
                fields[block], factors,
                lambda values, factor: solve_triangular(factor, values, lower=True))
        return float(np.sum(fields ** 2))
