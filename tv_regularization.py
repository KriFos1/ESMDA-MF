"""Spacing-aware, smoothed isotropic TV in natural-log resistivity coordinates.

The 2D penalty is integrated TV divided by window width, so an x-invariant
field has the same penalty as its 1D column. Outer edges have zero forward
differences; internal edges, including fixed/free interfaces, are retained.
"""

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


@lru_cache(maxsize=32)
def operators(shape, spacing):
    if len(shape) not in (1, 2) or len(shape) != len(spacing):
        raise ValueError('TV supports one or two spatial axes')
    if min(shape) < 1 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError('TV dimensions and spacings must be positive and finite')
    cells = np.arange(np.prod(shape)).reshape(shape)
    result = []
    for axis, step in enumerate(spacing):
        left = [slice(None)] * len(shape)
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        rows = cells[tuple(left)].ravel()
        columns = cells[tuple(right)].ravel()
        result.append(sparse.coo_matrix(
            (np.r_[-np.ones(len(rows)), np.ones(len(rows))] / step,
             (np.r_[rows, rows], np.r_[rows, columns])),
            shape=(cells.size, cells.size)).tocsr())
    cell_weight = spacing[0] if len(shape) == 1 else spacing[1] / shape[0]
    return tuple(result), cell_weight


def value_and_gradient(values, shape, spacing, epsilon=1e-3):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('TV epsilon must be finite and positive')
    values = np.asarray(values, dtype=float).reshape(-1)
    ops, cell_weight = operators(tuple(shape), tuple(spacing))
    gradients = [op @ values for op in ops]
    norms = np.sqrt(sum(gradient ** 2 for gradient in gradients) + epsilon ** 2)
    value = cell_weight * np.sum(norms - epsilon)
    derivative = cell_weight * sum(op.T @ (gradient / norms)
                                   for op, gradient in zip(ops, gradients))
    return float(value), np.asarray(derivative)


def irls_matrix(values, shape, spacing, epsilon):
    ops, cell_weight = operators(tuple(shape), tuple(spacing))
    gradients = [op @ values for op in ops]
    weights = cell_weight / np.sqrt(sum(gradient ** 2 for gradient in gradients) + epsilon ** 2)
    return sum(op.T @ sparse.diags(weights) @ op for op in ops).tocsr()


def prox(values, strength, spacing, epsilon=1e-3, iterations=8):
    """Euclidean TV proximal step; no forward model or Jacobian calls."""
    target = np.asarray(values, dtype=float)
    if not np.isfinite(strength) or strength < 0 or iterations < 1:
        raise ValueError('TV strength must be nonnegative and iterations positive')
    if strength == 0:
        return target.copy()
    current = target.ravel().copy()
    identity = sparse.eye(current.size, format='csc')
    for _ in range(iterations):
        matrix = identity + strength * irls_matrix(current, target.shape, spacing, epsilon)
        candidate = sparse.linalg.spsolve(matrix.tocsc(), target.ravel())
        if np.linalg.norm(candidate - current) <= 1e-6 * max(1., np.linalg.norm(current)):
            current = candidate
            break
        current = candidate
    return current.reshape(target.shape)


def regularize_columns(ensemble, strength, dz, epsilon=1e-3, iterations=8):
    """Regularize each member's Rh column independently; latent is untouched."""
    result = np.asarray(ensemble, dtype=float).copy()
    nz = result.shape[0] // 2
    if strength:
        for member in range(result.shape[1]):
            result[:nz, member] = prox(result[:nz, member], strength, (dz,), epsilon, iterations)
    return result


def regularize_windows(ensemble, nx, nz, strength, dx, dz, epsilon=1e-3, iterations=8):
    """Regularize each member's full 2D Rh window independently; latent untouched."""
    result = np.asarray(ensemble, dtype=float).copy()
    cells = nx * nz
    if strength:
        for member in range(result.shape[1]):
            result[:cells, member] = prox(result[:cells, member].reshape(nx, nz),
                                          strength, (dx, dz), epsilon, iterations).reshape(cells)
    return result


def linearized_step(parameters, prediction, mean, jacobian, observations, data_covariance,
                    shape, spacing, strength, active_mask, epsilon=1e-3, iterations=6,
                    amplitude_std=np.log(10.), max_step=.5, bounds=(-1., 4.)):
    """IRLS MAP-like step with a fixed Jacobian and weak independent prior.

    Only ln Rh has TV. Latent has an independent unit-variance amplitude
    penalty. Trust bounds constrain all active increments; inactive parameters
    remain fixed. This is a local linearized solve, not posterior sampling.
    """
    parameters = np.asarray(parameters, dtype=float)
    cells = int(np.prod(shape))
    if len(parameters) != 2 * cells or strength < 0 or amplitude_std <= 0 or max_step <= 0:
        raise ValueError('Invalid linearized TV configuration')
    active = np.flatnonzero(active_mask) if active_mask is not None else np.arange(len(parameters))
    if not len(active):
        return np.zeros_like(parameters)
    covariance = np.asarray(data_covariance)
    precision = np.r_[np.full(cells, amplitude_std ** -2), np.ones(cells)]
    # Small trust damping anchors the local model around its evaluation point.
    damping = .01
    weighted_jacobian = jacobian / covariance[:, None]
    hessian = jacobian.T @ weighted_jacobian + np.diag(precision + damping)
    rhs = weighted_jacobian.T @ (observations - prediction + jacobian @ parameters)
    rhs += precision * mean + damping * parameters
    current = parameters.copy()
    low, high = parameters[active] - max_step, parameters[active] + max_step
    rh_active = active < cells
    low[rh_active] = np.maximum(low[rh_active], bounds[0])
    high[rh_active] = np.minimum(high[rh_active], bounds[1])
    fixed = np.setdiff1d(np.arange(len(parameters)), active)
    for _ in range(iterations):
        penalty = sparse.block_diag((irls_matrix(current[:cells], shape, spacing, epsilon),
                                     sparse.csr_matrix((cells, cells)))).toarray()
        matrix = hessian + strength * penalty
        free_matrix = matrix[np.ix_(active, active)]
        free_rhs = rhs[active] - matrix[np.ix_(active, fixed)] @ parameters[fixed]
        scale = np.sqrt(np.maximum(np.diag(free_matrix), 1e-12))
        normalized = free_matrix / scale[:, None] / scale[None, :]
        normalized_rhs = free_rhs / scale

        def objective(z):
            product = normalized @ z
            return .5 * z @ product - normalized_rhs @ z, product - normalized_rhs

        solved = minimize(objective, current[active] * scale, jac=True, method='L-BFGS-B',
                          bounds=list(zip(low * scale, high * scale)),
                          options={'maxiter': 60, 'ftol': 1e-10})
        candidate = current.copy()
        candidate[active] = solved.x / scale
        if not np.isfinite(candidate).all():
            raise FloatingPointError('Non-finite TV linearized step')
        difference = np.linalg.norm(candidate - current)
        current = candidate
        if difference <= 1e-6 * max(1., np.linalg.norm(current)):
            break
    return current - parameters
