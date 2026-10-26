import json
from pathlib import Path

import numpy as np
import pytest

from matern_prior import MaternPrior, spde_operator
from rml_standalone import factor_apply, factor_solve, optimize_member_lm, randomized_inputs

ROOT = Path(__file__).resolve().parents[1]


def small_prior():
    return MaternPrior((6, 5), (5., 5.), [[20., 20.], [30., 12.]], [.8, 0.], [1.1, .12])


def test_operator_is_symmetric_sparse_and_well_conditioned():
    matrix, volume = spde_operator((33, 80), (5., 5.), (50., 50.))
    assert volume == 25. and abs(matrix - matrix.T).max() < 1e-12
    assert matrix.nnz <= 5 * matrix.shape[0]
    prior = MaternPrior((33, 80), (5., 5.), [[50., 50.], [80., 35.]], [.85, 0.], [1.09, .12])
    eigenvalues = np.linalg.eigvalsh(prior.blocks[0].precision().toarray())
    assert eigenvalues[0] > 0 and eigenvalues[-1] / eigenvalues[0] < 1e5


def test_factor_reproduces_precision_and_calibrated_marginal_std():
    prior = small_prior()
    factor = prior.color(np.eye(60))
    covariance = factor @ factor.T
    np.testing.assert_allclose(np.linalg.inv(prior.precision().toarray()), covariance, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), prior.std, rtol=1e-10)
    whitening = prior.whiten(np.eye(60))
    np.testing.assert_allclose(whitening.T @ whitening, prior.precision().toarray(), rtol=1e-8, atol=1e-10)


def test_whiten_color_and_transpose_are_consistent():
    prior = small_prior()
    rng = np.random.default_rng(3)
    z, y = rng.normal(size=(60, 4)), rng.normal(size=(60, 4))
    np.testing.assert_allclose(prior.whiten(prior.color(z)), z, atol=1e-10)
    np.testing.assert_allclose(np.sum(prior.color(z) * y), np.sum(z * prior.color(y, transpose=True)))
    np.testing.assert_allclose(factor_solve(prior, factor_apply(prior, z[:, 0])), z[:, 0], atol=1e-10)
    np.testing.assert_allclose(factor_apply(prior, y, transpose=True), prior.color(y, transpose=True))


def test_robin_boundary_limits_edge_variance_inflation():
    def base_variance(robin):
        matrix, volume = spde_operator((30, 30), (5., 5.), (50., 50.), robin=robin)
        inverse = np.linalg.inv(matrix.toarray())
        return (volume * np.sum(inverse ** 2, axis=1)).reshape(30, 30)
    neumann, robin = base_variance(False), base_variance(True)
    assert neumann[0, 0] / neumann[15, 15] > 2.
    assert abs(robin[0, 0] / robin[15, 15] - 1.) < abs(neumann[0, 0] / neumann[15, 15] - 1.)


def test_own_draws_are_typical_and_a_sharp_boundary_is_affordable():
    prior = MaternPrior((33, 80), (5., 5.), [[50., 50.], [80., 35.]], [.85, 0.], [1.09, .12])
    draws = prior.sample(200, 11)
    ratio = 2 * prior.penalty(draws, (0,)) / prior.cells
    assert abs(ratio.mean() - 1.) < .02
    step = np.r_[np.where(np.arange(80) < 40, .41, 3.40)[None, :].repeat(33, 0).ravel(), np.zeros(prior.cells)]
    assert 2 * prior.penalty(step, (0,))[0] / prior.cells < 5.


def test_lm_rml_with_matern_prior_reaches_exact_linear_gaussian_minimizer():
    prior = small_prior()
    rng = np.random.default_rng(28)
    matrix = rng.normal(size=(3, 60)) / 5
    observations, variance = np.array([.1, -.3, .2]), np.array([.02, .03, .04])
    inputs = randomized_inputs(prior, observations, variance, 2, 2045)
    covariance = prior.apply(np.eye(60))
    gain = covariance @ matrix.T @ np.linalg.inv(matrix @ covariance @ matrix.T + np.diag(variance))
    center, target = inputs['prior_centers'][0], inputs['targets'][0]
    fit = optimize_member_lm(prior, center, target, variance,
                             lambda models: [(matrix @ m, matrix) for m in models],
                             relative_gradient_tolerance=1e-10, stagnation_tolerance=1e-14,
                             max_iterations=300)
    np.testing.assert_allclose(fit['model'], center + gain @ (target - matrix @ center), atol=1e-6)


def benchmark_window_state(index=45):
    """Benchmark ln Rh and anisotropy latent upscaled to the index window (RML parameterization)."""
    h5py = pytest.importorskip('h5py')
    from scipy.special import ndtri
    from transition_mda_study import checkpoint, overlap_weights
    reference = ROOT / 'runs/benchmark3-full-two-step-esmda-b025-t75-20261015-r2'
    study_file = ROOT / 'runs/rml-index45-lm-20261019b/study.json'
    if not (reference / f'inversion_results_assim_{index}.pkl').exists() or not study_file.exists():
        pytest.skip('Benchmark reference run not available')
    manifest = json.loads((reference / 'run.json').read_text())
    benchmark = Path(manifest['benchmark']) / 'globalmodel.h5'
    if not benchmark.exists():
        pytest.skip('Benchmark model not available')
    study = json.loads(study_file.read_text())
    saved = checkpoint(reference, index)
    with h5py.File(benchmark, 'r') as handle:
        x, z = handle['formation/x'][:, 0] / .3048, handle['formation/z'][:, 0] / .3048
        middle = handle['formation/Rh'].shape[1] // 2
        rh, rv = handle['formation/Rh'][:, middle, :], handle['formation/Rv'][:, middle, :]
    nx, _, nz = saved['dims']
    dx, _, dz = saved['spacing_ft']
    shift = saved['shift_ft']
    wx = overlap_weights(x, shift['x'] + np.arange(nx + 1) * dx, float(np.median(np.diff(x))))
    wz = overlap_weights(z, shift['z'] + np.arange(nz + 1) * dz, float(np.median(np.diff(z))))
    ln_rh, ln_rv = np.log(wx @ rh.T @ wz.T), np.log(wx @ rv.T @ wz.T)
    fraction = (ln_rv - ln_rh - np.log(study['ratio_min'])) / np.log(study['ratio_max'] / study['ratio_min'])
    latent = ndtri(np.clip(fraction, 1e-6, 1 - 1e-6))
    return study, np.r_[ln_rh.ravel(), latent.ravel()]


def test_benchmark_model_is_plausible_under_the_matern_prior():
    """The index-45 benchmark ln Rh field (sharp layer boundaries) under the study's Matern prior.

    2 x penalty / n ~ 1 is a typical draw; much larger means the prior cannot represent
    the benchmark. The old squared-exponential prior scores ~1e16 on the same field.
    """
    import pilot_rml_index45 as runner
    study, state = benchmark_window_state()
    prior_spec = study['prior']
    prior = MaternPrior(prior_spec['shape'], prior_spec['spacing'], prior_spec['lengths'],
                        prior_spec['means'], prior_spec['stds'])
    rh_ratio = 2 * prior.penalty(state, (0,))[0] / prior.cells
    latent_ratio = 2 * prior.penalty(state, (1,))[0] / prior.cells
    squared_exponential = runner.load_prior(study)
    z = factor_solve(squared_exponential, state - squared_exponential.mean)
    se_ratio = np.sum(z[:prior.cells] ** 2) / prior.cells
    print(f'benchmark 2P/n: Matern ln Rh {rh_ratio:.3f}, Matern latent {latent_ratio:.1f}, '
          f'squared-exponential ln Rh {se_ratio:.3e}')
    assert .05 < rh_ratio < 3.                   # within the support, not an outlier
    assert se_ratio > 1e6 * rh_ratio             # the old prior effectively excludes it
    assert np.isfinite(latent_ratio)


def test_benchmark_anisotropy_is_plausible_with_log_clipped_parameterization():
    """ln(Rv/Rh) clipped at 0 (isotropy at finite values): Matern mean .5, std .5, ranges 80/35 ft."""
    study, state = benchmark_window_state()
    cells = state.size // 2
    ln_rh = state[:cells]
    ratio = study['ratio_min'] * (study['ratio_max'] / study['ratio_min']) ** __import__('scipy.special', fromlist=['ndtr']).ndtr(state[cells:])
    log_anisotropy = np.log(ratio)            # isotropic cells -> 0, an ordinary finite value
    spec = study['prior']
    prior = MaternPrior(spec['shape'], spec['spacing'], spec['lengths'], [spec['means'][0], .5],
                        [spec['stds'][0], .5])
    ratio_score = 2 * prior.penalty(np.r_[ln_rh, log_anisotropy], (1,))[0] / cells
    print(f'benchmark 2P/n, log-clipped anisotropy: {ratio_score:.3f}')
    assert .05 < ratio_score < 3.
