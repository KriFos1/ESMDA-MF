"""Gaussian increment prior with face weights: the conditionally Gaussian form of a Student-t increment prior.

Each state block (ln Rh, anisotropy latent) on the (nx, nz) window, C-order cell index ``ix * nz + iz``,
has the precision

    Q = Dz^T diag(wz) Dz + Dx^T diag(wx) Dx + diag(1 / anchor_std^2),

with Dz / Dx the differences between vertical / lateral neighbours and one weight per face. The increment
terms penalise the increments of the field itself and the anchor pulls each cell to ``anchor_mean``:

    0.5 m^T (Dz^T wz Dz + Dx^T wx Dx) m + 0.5 (m - anchor_mean)^T A (m - anchor_mean),  A = diag(1 / anchor_std^2),

which equals 0.5 (m - mean)^T Q (m - mean) + const with the Gaussian mean ``mean = Q^-1 A anchor_mean`` (the
anchor mean smoothed by the coupling). For fixed weights this is a Gaussian Markov random field with
spatially varying coupling; a small weight decouples two neighbours, which is where a boundary can form.
With the weights

    w = (dof + 1) / (dof * scale^2 + increment^2)

evaluated at a model (``student_weights``), minimising the Gaussian objective is one majorise-minimise
step for the Student-t increment prior (iteratively reweighted least squares).

The class has the factor interface of the other priors (``mean``, ``std``, ``shape``, ``color``,
``whiten``, ``sample``); the factor is a dense Cholesky factor per block, fine for windows of a few
thousand cells.
"""

import numpy as np
from scipy import sparse
from scipy.linalg import cholesky, solve_triangular


def face_differences(shape):
    """Dense difference matrices (vertical faces, lateral faces) of one block."""
    nx, nz = (int(n) for n in shape)
    index = np.arange(nx * nz).reshape(nx, nz)
    out = []
    for low, high in ((index[:, :-1].ravel(), index[:, 1:].ravel()), (index[:-1].ravel(), index[1:].ravel())):
        matrix = np.zeros((low.size, nx * nz))
        matrix[np.arange(low.size), high] = 1.
        matrix[np.arange(low.size), low] = -1.
        out.append(matrix)
    return out


def increments(model, shape):
    """Vertical and lateral increments of both blocks: arrays (2, nx, nz - 1) and (2, nx - 1, nz)."""
    fields = np.asarray(model, dtype=float).reshape(2, *shape)
    return np.diff(fields, axis=2), np.diff(fields, axis=1)


def student_weights(model, shape, dof, scales, strength=1.):
    """Face weights of the Student-t majoriser at ``model``.

    ``scales``: [[vertical, lateral] for ln Rh, [vertical, lateral] for the latent] (state units).
    ``strength`` multiplies the increment penalty (the exponent of every face density): below one it
    makes a boundary, which is a line of faces in 2D, cheaper relative to the data and the anchor.
    """
    scales = np.asarray(scales, dtype=float).reshape(2, 2)
    vertical, lateral = increments(model, shape)
    return (strength * (dof + 1.) / (dof * scales[:, 0, None, None] ** 2 + vertical ** 2),
            strength * (dof + 1.) / (dof * scales[:, 1, None, None] ** 2 + lateral ** 2))


def student_penalty(model, shape, dof, scales, mean, anchor_std, strength=1.):
    """Negative log of the Student-t increment prior with its Gaussian anchor (up to a constant)."""
    scales = np.asarray(scales, dtype=float).reshape(2, 2)
    vertical, lateral = increments(model, shape)
    tails = (np.sum(np.log1p(vertical ** 2 / (dof * scales[:, 0, None, None] ** 2)))
             + np.sum(np.log1p(lateral ** 2 / (dof * scales[:, 1, None, None] ** 2))))
    return .5 * strength * (dof + 1.) * tails + .5 * np.sum(((np.asarray(model) - mean) / anchor_std) ** 2)


def ensemble_student_weights(state, shape, dof, scales, statistic='members', strength=1.):
    """Face weights of the Student-t majoriser for an ensemble ``state`` (2 * cells, members).

    ``statistic`` selects the squared increment the weight is evaluated at:
      'mean'     the increment of the ensemble mean: a boundary is where the mean has one
      'members'  the member average of the squared increments: faces where the members disagree about a
                 boundary are loosened as well, which keeps spread where its position is uncertain
    """
    state = np.asarray(state, dtype=float)
    scales = np.asarray(scales, dtype=float).reshape(2, 2)
    fields = state.reshape(2, *shape, -1)
    if statistic == 'mean':
        fields = fields.mean(axis=3, keepdims=True)
    elif statistic != 'members':
        raise ValueError(f'Unknown weight statistic {statistic}')
    vertical = np.mean(np.diff(fields, axis=2) ** 2, axis=3)
    lateral = np.mean(np.diff(fields, axis=1) ** 2, axis=3)
    return (strength * (dof + 1.) / (dof * scales[:, 0, None, None] ** 2 + vertical),
            strength * (dof + 1.) / (dof * scales[:, 1, None, None] ** 2 + lateral))


def increment_precision(shape, weights_vertical, weights_lateral, anchor_std):
    """Sparse precision of both blocks: Dz^T wz Dz + Dx^T wx Dx + diag(1 / anchor_std^2) per block.

    ``anchor_std``: one value per block or per state entry. Same matrix as IncrementPrior builds densely.
    """
    nx, nz = (int(n) for n in shape)
    cells = nx * nz
    index = np.arange(cells).reshape(nx, nz)
    anchor = np.asarray(anchor_std, dtype=float)
    anchor = np.repeat(anchor, cells) if anchor.size == 2 else anchor.ravel()
    weights = (np.asarray(weights_vertical, dtype=float).reshape(2, nx, nz - 1),
               np.asarray(weights_lateral, dtype=float).reshape(2, nx - 1, nz))
    if anchor.shape != (2 * cells,) or np.any(anchor <= 0) or any(np.any(w <= 0) for w in weights):
        raise ValueError('Require positive anchor std and positive face weights')
    faces = ((index[:, :-1].ravel(), index[:, 1:].ravel()), (index[:-1].ravel(), index[1:].ravel()))
    blocks = []
    for block in range(2):
        precision = sparse.diags_array(1. / anchor[block * cells:(block + 1) * cells] ** 2, format='csr')
        for (low, high), weight in zip(faces, weights):
            rows = np.arange(low.size)
            difference = sparse.csr_array((np.concatenate([np.ones(low.size), -np.ones(low.size)]),
                                           (np.concatenate([rows, rows]), np.concatenate([high, low]))),
                                          shape=(low.size, cells))
            precision = precision + difference.T @ sparse.diags_array(weight[block].ravel()) @ difference
        blocks.append(precision)
    return sparse.block_diag(blocks, format='csr')


class IncrementPrior:
    def __init__(self, anchor_mean, anchor_std, weights_vertical, weights_lateral, shape):
        nx, nz = (int(n) for n in shape)
        self.shape = (2, nx, nz)
        self.cells = nx * nz
        self.anchor_mean = np.asarray(anchor_mean, dtype=float).ravel().copy()
        self.anchor_std = np.asarray(anchor_std, dtype=float).ravel().copy()
        self.weights_vertical = np.asarray(weights_vertical, dtype=float).reshape(2, nx, nz - 1).copy()
        self.weights_lateral = np.asarray(weights_lateral, dtype=float).reshape(2, nx - 1, nz).copy()
        if (self.anchor_mean.shape != (2 * self.cells,) or self.anchor_std.shape != self.anchor_mean.shape
                or np.any(self.anchor_std <= 0) or np.any(self.weights_vertical <= 0)
                or np.any(self.weights_lateral <= 0)):
            raise ValueError('Require one mean and positive anchor std per state entry and positive face weights')
        d_vertical, d_lateral = face_differences((nx, nz))
        self.inverse_factors, std, mean = [], [], []
        for block in range(2):
            rows = slice(block * self.cells, (block + 1) * self.cells)
            precision = (d_vertical.T @ (self.weights_vertical[block].ravel()[:, None] * d_vertical)
                         + d_lateral.T @ (self.weights_lateral[block].ravel()[:, None] * d_lateral)
                         + np.diag(1. / self.anchor_std[rows] ** 2))
            factor = cholesky(precision, lower=True)                     # Q = R R^T
            inverse = solve_triangular(factor, np.eye(self.cells), lower=True)   # R^-1; C = R^-T R^-1
            self.inverse_factors.append((factor, inverse))
            std.append(np.sqrt(np.sum(inverse ** 2, axis=0)))
            # Gaussian mean Q^-1 A anchor_mean: the increments of the field, not of its offset, are penalised.
            mean.append(inverse.T @ (inverse @ (self.anchor_mean[rows] / self.anchor_std[rows] ** 2)))
        self.std, self.mean = np.concatenate(std), np.concatenate(mean)

    def _blockwise(self, values, operation):
        values = np.asarray(values, dtype=float)
        vector = values.ndim == 1
        values = values.reshape(2 * self.cells, -1)
        result = np.vstack([operation(*self.inverse_factors[block],
                                      values[block * self.cells:(block + 1) * self.cells])
                            for block in range(2)])
        return result[:, 0] if vector else result

    def color(self, values, transpose=False):
        """L z or L^T y with L = R^-T, so that L L^T = Q^-1."""
        if transpose:
            return self._blockwise(values, lambda factor, inverse, part: inverse @ part)
        return self._blockwise(values, lambda factor, inverse, part: inverse.T @ part)

    def whiten(self, offsets):
        """L^-1 (m - mean) = R^T (m - mean)."""
        return self._blockwise(offsets, lambda factor, inverse, part: factor.T @ part)

    def sample(self, count, seed):
        noise = np.random.default_rng(seed).normal(size=(2 * self.cells, count))
        return self.mean[:, None] + self.color(noise)

    def save(self, path):
        np.savez_compressed(path, anchor_mean=self.anchor_mean, anchor_std=self.anchor_std,
                            weights_vertical=self.weights_vertical, weights_lateral=self.weights_lateral,
                            shape=np.array(self.shape[1:]))

    @classmethod
    def load(cls, path):
        with np.load(path) as saved:
            return cls(saved['anchor_mean'], saved['anchor_std'], saved['weights_vertical'], saved['weights_lateral'],
                       tuple(int(n) for n in saved['shape']))
