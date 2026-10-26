"""Batched independent randomized maximum likelihood and Gaussian moment transfer."""

import numpy as np
from geostat.gaussian_sim import fast_gaussian

from lm_laplace import optimize_batch


def run_rml(ensemble, predictions_1d, observations, variance, prior,
            forward_batch, derivative_batch, rng, count=20, **options):
    scores = np.mean((predictions_1d - observations[:, None]) ** 2 / variance[:, None], axis=0)
    selected = np.argsort(scores, kind='stable')[:count]
    if len(selected) != count:
        raise ValueError('RML needs the requested number of distinct ensemble members')
    initials = ensemble[:, selected].T.copy()
    prior_centers = (prior.mean[:, None] + prior.draw(count, rng)).T
    noise = rng.normal(size=(count, len(observations))) * np.sqrt(variance)[None, :]
    targets = observations[None, :] + noise
    initial_predictions = forward_batch(list(initials))
    fits = optimize_batch(
        list(initials), initial_predictions, prior, observations, variance,
        forward_batch, derivative_batch, prior_means=prior_centers, datums=targets,
        final_jacobian=False, **options)
    usable = np.array([
        fit['reason'] in ('relative_data_improvement', 'iteration_limit', 'no_acceptable_trial')
        and len(fit['objective_history']) > 1 and np.isfinite(fit['model']).all()
        and np.isfinite(fit['prediction']).all()
        for fit in fits])
    return {'fits': fits, 'usable': usable, 'selected': selected, 'scores_1d': scores,
            'prior_centers': prior_centers.T, 'targets': targets.T,
            'measurement_noise': noise.T, 'initials': initials.T}


def gaussian_moment_redraw(models, shape, spacing, lengths, count, seed):
    """Match fitted state-coordinate moments, with the established spatial kernels."""
    nx, nz = shape
    if models.shape[0] != 2 * nx * nz or models.shape[1] < 2:
        raise ValueError('Posterior models must contain both state blocks and at least two members')
    mean = models.mean(axis=1)
    std = models.std(axis=1, ddof=1)
    constant = np.ptp(models, axis=1) == 0
    mean[constant] = models[constant, 0]
    std[constant] = 0.
    # fast_gaussian uses numpy's legacy RNG; keep it isolated from the rest of the run.
    previous_rng = np.random.get_state()
    try:
        np.random.seed(seed)
        rh = fast_gaussian(np.array([nz, nx]), np.array([1.]),
                           np.array([lengths[0][1] / spacing[1], lengths[0][0] / spacing[0]]),
                           num_samples=count)
    finally:
        np.random.set_state(previous_rng)
    from run_DA_sequential import _ar1_field
    previous_rng = np.random.get_state()
    try:
        np.random.seed(seed + 1)
        latent = _ar1_field(nx, nz, count, np.exp(-spacing[0] / lengths[1][0]),
                            np.exp(-spacing[1] / lengths[1][1])).reshape(nx * nz, count)
    finally:
        np.random.set_state(previous_rng)
    innovations = np.vstack([rh, latent])
    innovations -= innovations.mean(axis=1, keepdims=True)
    innovations /= innovations.std(axis=1, ddof=1, keepdims=True)
    return mean[:, None] + std[:, None] * innovations, mean, std
