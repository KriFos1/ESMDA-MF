"""Matern (nu = 1, alpha = 2) SPDE prior with a sparse Markov precision on the 2D window.

Each state block (ln Rh, anisotropy latent) on the (nx, nz) window, C-order
cell index ``ix * nz + iz``, is a Gaussian Markov random field defined by the
SPDE (kappa^2 - div H grad) x = W of Lindgren, Rue and Lindstrom (2011),
following the conventions of matern-egrid / JutulDarcy:

* practical ranges (rho_x, rho_z); rho = sqrt(rho_x rho_z), kappa = sqrt(8 nu) / rho,
  H = diag((rho_x / rho)^2, (rho_z / rho)^2), two-point flux finite volumes;
* K = L_H + diag(V kappa^2 + robin), robin = sum(boundary face length) kappa / 1.42;
* mass matrix C = V I (uniform cells), base precision K C^-1 K (alpha = 2: nu = 1 in 2D).

The marginal standard deviation is calibrated cellwise, S = sigma / sqrt(diag(K^-1 C K^-1)),
so edge cells keep the target spread. The model offset is m - mu = L z with
L = S K^-1 C^1/2: whitening z = C^-1/2 K S^-1 (m - mu) and the prior penalty
0.5 |z|^2 = 0.5 (m - mu)^T Q (m - mu), Q = S^-1 K C^-1 K S^-1, need only sparse
products. Coloring is one sparse LU solve with K.
"""

import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

ROBIN_SCALE = 1.42
NU = 1.0


def spde_operator(shape, spacing, ranges, robin=True):
    """Sparse K (CSC) and the cell volume for one block on a regular (nx, nz) grid."""
    nx, nz = (int(n) for n in shape)
    dx, dz = (float(h) for h in spacing)
    range_x, range_z = (float(r) for r in ranges)
    if min(nx, nz) < 2 or min(dx, dz, range_x, range_z) <= 0:
        raise ValueError('Require a 2D grid with positive spacing and ranges')
    rho = math.sqrt(range_x * range_z)
    kappa = math.sqrt(8. * NU) / rho
    lam_x, lam_z = (range_x / rho) ** 2, (range_z / rho) ** 2
    volume = dx * dz
    index = np.arange(nx * nz).reshape(nx, nz)
    diagonal = np.full(nx * nz, volume * kappa ** 2)
    rows, cols, values = [], [], []
    # Interior faces: x-faces have length dz and centre distance dx (and vice versa).
    for left, right, transmissibility in ((index[:-1].ravel(), index[1:].ravel(), lam_x * dz / dx),
                                          (index[:, :-1].ravel(), index[:, 1:].ravel(), lam_z * dx / dz)):
        np.add.at(diagonal, left, transmissibility)
        np.add.at(diagonal, right, transmissibility)
        rows += [left, right]
        cols += [right, left]
        values += [np.full(left.size, -transmissibility)] * 2
    if robin:
        for cells, length in ((index[0], dz), (index[-1], dz), (index[:, 0], dx), (index[:, -1], dx)):
            np.add.at(diagonal, cells.ravel(), length * kappa / ROBIN_SCALE)
    matrix = sp.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(nx * nz, nx * nz)) + sp.diags(diagonal)
    return matrix.tocsc(), volume


class MaternBlock:
    """One Matern GMRF block with cellwise-calibrated marginal standard deviation."""

    def __init__(self, shape, spacing, ranges, mean, std, robin=True):
        self.K, self.volume = spde_operator(shape, spacing, ranges, robin)
        self.n = self.K.shape[0]
        self.lu = splu(self.K)
        # Base field x = K^-1 C^1/2 xi has covariance V K^-2 (K symmetric).
        inverse = self.lu.solve(np.eye(self.n))
        base_variance = self.volume * np.sum(inverse ** 2, axis=1)
        self.mean = np.broadcast_to(np.asarray(mean, dtype=float), (self.n,)).copy()
        self.std = np.broadcast_to(np.asarray(std, dtype=float), (self.n,)).copy()
        if np.any(self.std <= 0):
            raise ValueError('Marginal standard deviation must be positive')
        self.scale = self.std / np.sqrt(base_variance)

    def color(self, values, transpose=False):
        """L z (or L^T y): L = S K^-1 C^1/2, K symmetric."""
        root = math.sqrt(self.volume)
        if transpose:
            return root * self.lu.solve(self.scale[:, None] * values)
        return self.scale[:, None] * self.lu.solve(root * values)

    def whiten(self, offset):
        """L^-1 (m - mu) = C^-1/2 K S^-1 (m - mu)."""
        return (self.K @ (offset / self.scale[:, None])) / math.sqrt(self.volume)

    def precision(self):
        inverse_scale = sp.diags(1. / self.scale)
        return (inverse_scale @ self.K @ self.K @ inverse_scale / self.volume).tocsr()

    def condition_entered(self, values, draws, entered):
        """Replace the entered cells of each column by draws conditional on the kept cells.

        For x ~ N(mu, Q^-1) split into entered (A) and kept (B) cells,
        x_A | x_B ~ N(mu_A - Q_AA^-1 Q_AB (x_B - mu_B), Q_AA^-1). Conditioning an
        unconditional draw x* by kriging gives x_A^c = x*_A + Q_AA^-1 Q_AB (x*_B - x_B),
        so only sparse solves with the entered-cell precision block are needed; by
        the Markov property only kept cells within the stencil of A contribute.
        """
        kept = ~entered
        precision = self.precision()
        block = precision[entered][:, entered].tocsc()
        coupling = precision[entered][:, kept]
        result = np.array(values, dtype=float, copy=True)
        result[entered] = draws[entered] + splu(block).solve(coupling @ (draws[kept] - values[kept]))
        return result


class MaternPrior:
    """Two-block (ln Rh, latent) Matern prior with the samplers' factor interface.

    Exposes ``mean``, ``std`` and ``shape`` like the other priors, and
    ``color`` / ``whiten`` used by ``rml_standalone.factor_apply`` / ``factor_solve``.
    """

    def __init__(self, shape, spacing, ranges, means, stds, robin=True):
        self.shape = (2, *(int(n) for n in shape))
        if len(ranges) != 2 or len(means) != 2 or len(stds) != 2:
            raise ValueError('Provide ranges, means and stds for both blocks')
        self.blocks = [MaternBlock(shape, spacing, block_ranges, mean, std, robin)
                       for block_ranges, mean, std in zip(ranges, means, stds)]
        self.mean = np.concatenate([block.mean for block in self.blocks])
        self.std = np.concatenate([block.std for block in self.blocks])
        self.cells = self.blocks[0].n

    def _blockwise(self, values, operation):
        values = np.asarray(values, dtype=float)
        vector = values.ndim == 1
        values = values.reshape(2 * self.cells, -1)
        result = np.vstack([operation(block, values[index * self.cells:(index + 1) * self.cells])
                            for index, block in enumerate(self.blocks)])
        return result[:, 0] if vector else result

    def color(self, values, transpose=False):
        return self._blockwise(values, lambda block, part: block.color(part, transpose))

    def whiten(self, offsets):
        return self._blockwise(offsets, lambda block, part: block.whiten(part))

    def sample(self, count, seed):
        noise = np.random.default_rng(seed).normal(size=(2 * self.cells, count))
        return self.mean[:, None] + self.color(noise)

    def penalty(self, models, blocks=(0, 1)):
        """0.5 (m - mu)^T Q (m - mu), per column, summed over the requested blocks."""
        whitened = self.whiten(np.asarray(models, dtype=float).reshape(2 * self.cells, -1)
                               - self.mean[:, None])
        total = sum(np.sum(whitened[index * self.cells:(index + 1) * self.cells] ** 2, axis=0)
                    for index in blocks)
        return .5 * total

    def precision(self):
        return sp.block_diag([block.precision() for block in self.blocks]).tocsr()

    def condition_entered(self, state, entered, seed):
        """Redraw the entered window cells of every member conditionally on the kept cells.

        ``entered`` is a boolean cell mask shared by both blocks; the kept cells of
        ``state`` (2 * cells, members) are left untouched. The conditional draws use
        the prior GMRF, so entered cells get prior conditional spread and the prior
        correlation with the neighbouring (assimilated) kept cells.
        """
        state = np.array(state, dtype=float, copy=True).reshape(2 * self.cells, -1)
        entered = np.asarray(entered, dtype=bool).ravel()
        if entered.shape != (self.cells,):
            raise ValueError('Entered-cell mask must cover the window once')
        if not entered.any():
            return state
        draws = self.sample(state.shape[1], seed)
        if entered.all():
            return draws
        for index, block in enumerate(self.blocks):
            rows = slice(index * self.cells, (index + 1) * self.cells)
            state[rows] = block.condition_entered(state[rows], draws[rows], entered)
        return state

    def apply(self, values):
        """Covariance product C v = L L^T v (interface parity with SpatialPrior)."""
        return self.color(self.color(values, transpose=True))


def from_study_prior(prior):
    """Build from a study's prior dictionary: lengths are practical ranges [[x, z], [x, z]]."""
    return MaternPrior(tuple(prior['shape']), tuple(prior['spacing']), prior['lengths'],
                       prior['means'], prior['stds'], robin=prior.get('boundary', 'robin') == 'robin')
