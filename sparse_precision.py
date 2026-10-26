"""Sparse-precision Gaussian prior estimated from an ensemble by local ridge regressions.

A modified Cholesky (Vecchia-type) estimate. The state's variables are ordered
cell by cell in raster order (x outer, z inner), each cell's ln Rh before its
anisotropy latent. Each variable's ensemble anomaly is regressed, with a ridge
penalty chosen by generalized cross-validation, on the anomalies of the
variables that precede it within a (2r + 1) x (2r + 1) cell neighbourhood:

    x_i - mu_i = sum_j B_ij (x_j - mu_j) + e_i,    e_i ~ N(0, d_i).

The precision Q = (I - B)^T D^-1 (I - B) is sparse and positive definite, and
the triangular factor T = I - B gives sampling and whitening by sparse
triangular solves: C = T^-1 D T^-T. Marginal standard deviations below the
per-block floors are lifted by a cellwise scaling S (correlations unchanged), so
the prior covariance is S C S, with

    color(z) = S P^T T^-1 D^1/2 z,    whiten(x) = D^-1/2 T P S^-1 x,

where P permutes the state into the regression order.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

RIDGE_GRID = np.logspace(-6., 2., 33)   # penalty relative to the largest squared singular value
STD_FLOORS = (.23, .1)                  # ln Rh, latent: the chain's earlier fitted-prior floors


def regression_order(nx, nz):
    """State indices in regression order: cells in raster order, ln Rh before latent."""
    cells = nx * nz
    cell = np.arange(cells)
    return np.column_stack([cell, cells + cell]).ravel()


def predecessors(nx, nz, radius):
    """Regression order and, per position, the earlier positions within the cell neighbourhood."""
    cells = nx * nz
    order = regression_order(nx, nz)
    position = np.empty(2 * cells, dtype=int)
    position[order] = np.arange(2 * cells)
    earlier = []
    for p, state_index in enumerate(order):
        x, z = divmod(state_index % cells, nz)
        xs = np.arange(max(x - radius, 0), min(x + radius, nx - 1) + 1)
        zs = np.arange(max(z - radius, 0), min(z + radius, nz - 1) + 1)
        neighbours = (xs[:, None] * nz + zs[None, :]).ravel()
        candidates = position[np.concatenate([neighbours, cells + neighbours])]
        earlier.append(np.sort(candidates[candidates < p]))
    return order, earlier


def ridge_gcv(predictors, target):
    """Ridge regression of a centred target on centred predictors, penalty by GCV.

    Predictors are standardized for the penalty. Returns the coefficients (in
    the original units), the residual variance RSS / (n - 1 - dof) and the
    chosen penalty relative to the largest squared singular value.
    """
    predictors = np.asarray(predictors, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(target)
    coefficients = np.zeros(predictors.shape[1])
    scale = predictors.std(axis=0, ddof=1) if predictors.shape[1] else np.zeros(0)
    keep = scale > 1e-12 * max(float(np.abs(target).max(initial=0.)), 1e-300)
    if not keep.any():
        return coefficients, float(target @ target / (n - 1)), np.nan
    u, s, vt = np.linalg.svd(predictors[:, keep] / scale[keep], full_matrices=False)
    projection = u.T @ target
    outside = max(float(target @ target - projection @ projection), 0.)
    squares = s ** 2
    penalties = squares.max() * RIDGE_GRID
    shrink = squares[None, :] / (squares[None, :] + penalties[:, None])
    rss = outside + np.sum(((1. - shrink) * projection[None, :]) ** 2, axis=1)
    dof = shrink.sum(axis=1)
    gcv = n * rss / np.maximum(n - dof, 1e-9) ** 2
    best = int(np.argmin(gcv))
    coefficients[keep] = (vt.T @ (shrink[best] / s * projection)) / scale[keep]
    return coefficients, float(rss[best] / max(n - 1. - dof[best], 1.)), float(RIDGE_GRID[best])


def conditional_regression(precision, observed):
    """Gaussian regression of the other variables on ``observed`` under a sparse precision Q.

    Returns the indices of the other variables and a function that maps
    increments of the observed variables (rows ``observed``, any number of
    columns) to the conditional-mean increments of the others,
    -Q_rr^-1 Q_ro dx_o, which equals C_ro C_oo^-1 dx_o. For data that see only
    the observed variables this is the exact Kalman update of the rest, and it
    does not change as such data are assimilated.
    """
    precision = sparse.csr_matrix(precision)
    n = precision.shape[0]
    observed = np.asarray(observed, dtype=int)
    rest = np.setdiff1d(np.arange(n), observed)
    factor = splu(precision[rest][:, rest].tocsc())
    coupling = precision[rest][:, observed].tocsc()

    def propagate(increment):
        increment = np.asarray(increment, dtype=float)
        vector = increment.ndim == 1
        result = -factor.solve(np.asarray(coupling @ increment.reshape(len(observed), -1)))
        return result[:, 0] if vector else result

    return rest, propagate


class EnsemblePrecisionPrior:
    """Gaussian prior N(mean, S C S) with the sparse precision fitted to an ensemble.

    Exposes ``mean``, ``std``, ``shape``, ``color`` and ``whiten`` like the
    other priors of the LM and Laplace code (``rml_standalone.factor_apply``).
    """

    def __init__(self, mean, order, factor, conditional_variance, scale, std, shape, radius, floors,
                 penalties=None):
        self.mean = np.asarray(mean, dtype=float)
        self.order = np.asarray(order, dtype=int)
        self.factor = sparse.csr_matrix(factor)
        self.conditional_variance = np.asarray(conditional_variance, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.shape = tuple(int(n) for n in shape)
        self.radius = int(radius)
        self.floors = tuple(float(f) for f in floors)
        self.penalties = None if penalties is None else np.asarray(penalties, dtype=float)
        self._root = np.sqrt(self.conditional_variance)
        self._scale_position = self.scale[self.order]
        self._lu = splu(self.factor.tocsc(), permc_spec='NATURAL', diag_pivot_thresh=0.)

    @classmethod
    def fit(cls, ensemble, shape, radius=2, floors=STD_FLOORS):
        """Fit to an ensemble of shape (2 * nx * nz, members); ``shape`` = (nx, nz)."""
        ensemble = np.asarray(ensemble, dtype=float)
        nx, nz = (int(n) for n in shape)
        n = 2 * nx * nz
        if ensemble.ndim != 2 or ensemble.shape[0] != n or ensemble.shape[1] < 3:
            raise ValueError('Ensemble must have shape (2 * nx * nz, members >= 3)')
        if radius < 1:
            raise ValueError('Neighbourhood radius must be at least one cell')
        mean = ensemble.mean(axis=1)
        order, earlier = predecessors(nx, nz, radius)
        anomalies = (ensemble - mean[:, None])[order].T          # members x positions
        rows, cols, values = [], [], []
        variance = np.empty(n)
        penalties = np.full(n, np.nan)
        for p in range(n):
            rows.append(p)
            cols.append(p)
            values.append(1.)
            if earlier[p].size:
                beta, variance[p], penalties[p] = ridge_gcv(anomalies[:, earlier[p]], anomalies[:, p])
                rows.extend([p] * len(earlier[p]))
                cols.extend(earlier[p])
                values.extend(-beta)
            else:
                variance[p] = float(anomalies[:, p] @ anomalies[:, p] / (len(anomalies) - 1))
        # Conditional variances stay positive even for a constant (clamped) variable.
        variance = np.maximum(variance, 1e-12 * max(float(variance.max()), 1e-300))
        factor = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
        unscaled = cls(mean, order, factor, variance, np.ones(n), np.ones(n), (2, nx, nz), radius, floors)
        marginal = np.sqrt(unscaled.marginal_variance())
        floor = np.where(np.arange(n) < nx * nz, floors[0], floors[1])
        scale = np.maximum(1., floor / marginal)
        return cls(mean, order, factor, variance, scale, scale * marginal, (2, nx, nz), radius, floors,
                   penalties)

    def marginal_variance(self, chunk=512):
        """Diagonal of S C S, accumulated over column chunks of T^-1 D^1/2."""
        n = len(self.mean)
        total = np.zeros(n)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            rhs = np.zeros((n, stop - start))
            rhs[np.arange(start, stop), np.arange(stop - start)] = self._root[start:stop]
            total += np.sum(self._lu.solve(rhs) ** 2, axis=1)
        result = np.empty(n)
        result[self.order] = total * self._scale_position ** 2
        return result

    def color(self, values, transpose=False):
        """L z with L L^T the prior covariance (or L^T v with ``transpose``)."""
        values = np.asarray(values, dtype=float)
        vector = values.ndim == 1
        values = values.reshape(len(self.mean), -1)
        if transpose:
            result = self._lu.solve(values[self.order] * self._scale_position[:, None], trans='T') \
                * self._root[:, None]
        else:
            solved = self._lu.solve(values * self._root[:, None])
            result = np.empty_like(solved)
            result[self.order] = solved * self._scale_position[:, None]
        return result[:, 0] if vector else result

    def whiten(self, offsets):
        """L^-1 (x - mean) for model offsets x - mean."""
        offsets = np.asarray(offsets, dtype=float)
        vector = offsets.ndim == 1
        values = offsets.reshape(len(self.mean), -1)
        result = (self.factor @ (values[self.order] / self._scale_position[:, None])) / self._root[:, None]
        return result[:, 0] if vector else result

    def sample(self, count, seed):
        rng = np.random.default_rng(seed)
        return self.mean[:, None] + self.color(rng.normal(size=(len(self.mean), count)))

    def precision(self):
        """Sparse precision of the floored prior: S^-1 P^T T^T D^-1 T P S^-1 (state order)."""
        permutation = sparse.csr_matrix((np.ones(len(self.order)), (np.arange(len(self.order)), self.order)))
        inner = self.factor.T @ sparse.diags(1. / self.conditional_variance) @ self.factor
        unscale = sparse.diags(1. / self.scale)
        return (unscale @ permutation.T @ inner @ permutation @ unscale).tocsr()

    def summary(self):
        neighbours = np.diff(self.factor.indptr) - 1
        return {'radius': self.radius, 'floors': list(self.floors),
                'mean_predictors': float(neighbours.mean()), 'max_predictors': int(neighbours.max()),
                'nonzeros_factor': int(self.factor.nnz),
                'median_relative_penalty': (float(np.nanmedian(self.penalties))
                                            if self.penalties is not None else None),
                'floored_fraction': float(np.mean(self.scale > 1.)),
                'median_std_ln_rh': float(np.median(self.std[:len(self.std) // 2])),
                'median_std_latent': float(np.median(self.std[len(self.std) // 2:]))}

    def save(self, path):
        factor = self.factor.tocsr()
        with open(path, 'xb') as stream:
            np.savez(stream, mean=self.mean, order=self.order, data=factor.data, indices=factor.indices,
                     indptr=factor.indptr, conditional_variance=self.conditional_variance, scale=self.scale,
                     std=self.std, shape=np.asarray(self.shape), radius=np.int64(self.radius),
                     floors=np.asarray(self.floors),
                     penalties=self.penalties if self.penalties is not None else np.zeros(0))

    @classmethod
    def load(cls, path):
        with np.load(path) as saved:
            n = len(saved['mean'])
            factor = sparse.csr_matrix((saved['data'], saved['indices'], saved['indptr']), shape=(n, n))
            penalties = saved['penalties'] if saved['penalties'].size else None
            return cls(saved['mean'], saved['order'], factor, saved['conditional_variance'], saved['scale'],
                       saved['std'], tuple(saved['shape']), int(saved['radius']), tuple(saved['floors']),
                       penalties)
