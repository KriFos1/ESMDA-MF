import importlib

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('pipt.update_schemes.analysis.enif', reason='EnIF-MDA needs PET >= 0.3 (venv-pet-latest)')

from enif_mda import EnIFProbe, active_rows


def chain_precision(n, coupling=.45):
    """A sparse SPD precision: a first-order chain."""
    off = np.full(n - 1, -coupling)
    return sparse.csc_array(sparse.diags([off, np.full(n, 1.), off], [-1, 0, 1]))


def problem(members=400, seed=3):
    rng = np.random.default_rng(seed)
    precision = chain_precision(8)
    root = np.linalg.cholesky(np.linalg.inv(precision.toarray()))
    state = 2. + root @ rng.normal(size=(8, members))
    sensitivity = np.zeros((2, 8))
    sensitivity[0, 2], sensitivity[1, 5] = 1., -.8
    return precision, state, sensitivity


def test_seeded_matern_precision_replaces_the_fit_and_is_carried(monkeypatch):
    enif_module = importlib.import_module('pipt.update_schemes.analysis.enif')
    monkeypatch.setattr(enif_module, 'fit_precision_cholesky_approximate',
                        lambda *args, **kwargs: pytest.fail('the seeded precision must replace the fit'))
    precision, state, sensitivity = problem()
    observations, variance = np.array([2.5, -1.2]), np.array([.05, .05])
    probe = EnIFProbe(precision, observations, variance, [2., 2.], seed=11)
    rng = np.random.default_rng(5)
    for iteration in range(2):
        predicted = sensitivity @ state
        perturbed = observations[:, None] + np.sqrt(2. * variance)[:, None] * rng.normal(size=predicted.shape)
        if iteration == 0:
            scales = state.std(axis=1)
        state = state + probe.step(state, predicted, perturbed)
        if iteration == 0:
            # Step 0 used the seeded precision in the ensemble's standardised coordinates.
            np.testing.assert_allclose(probe.analysis.Prec_u.toarray(),
                                       np.diag(scales) @ precision.toarray() @ np.diag(scales), atol=1e-12)
        assert probe.scheme.iteration == iteration + 1 == probe.scheme.enif_information.iteration
    # Two half-weight steps on a linear model approach the Gaussian posterior mean.
    prior_cov = np.linalg.inv(precision.toarray())
    _, prior_state, _ = problem()
    mean = prior_state.mean(axis=1)
    gain = prior_cov @ sensitivity.T @ np.linalg.inv(sensitivity @ prior_cov @ sensitivity.T + np.diag(variance))
    exact = mean + gain @ (observations - sensitivity @ mean)
    assert np.abs(state.mean(axis=1) - exact).max() < .15


def test_rows_that_lose_spread_are_held_fixed_and_conditioned_out():
    precision, state, sensitivity = problem()
    observations, variance = np.array([2.5, -1.2]), np.array([.05, .05])
    probe = EnIFProbe(precision, observations, variance, [2., 2.], seed=11)
    rng = np.random.default_rng(6)
    predicted = sensitivity @ state
    state = state + probe.step(state, predicted, observations[:, None] + .3 * rng.normal(size=predicted.shape))
    state[7] = 1.5                                               # e.g. every member clamped to a bound
    assert list(active_rows(state)) == list(range(7))
    predicted = sensitivity @ state
    step = probe.step(state, predicted, observations[:, None] + .3 * rng.normal(size=predicted.shape))
    np.testing.assert_array_equal(step[7], 0.)
    assert list(probe.scheme.enif_information.active_rows) == list(range(7))
    state[7] = rng.normal(size=state.shape[1])                   # spread cannot come back
    with pytest.raises(ValueError, match='regained'):
        probe.step(state, sensitivity @ state, np.broadcast_to(observations[:, None], (2, state.shape[1])))


def test_driver_runs_the_gated_chain_with_enif(tmp_path, gate_environment):
    import json
    import chain_laplace
    import run_chain_laplace_loop as loop
    import sequential_matern_1d as driver
    _, reference = gate_environment
    output = tmp_path / 'enif'
    assert driver.main(['--reference', str(reference), '--output-dir', str(output), '--stop', '4',
                        '--ensemble-size', '40', '--mda-steps', '3', '--memory-ft', '20', '--gate-threshold', '10',
                        '--analysis', 'enif', '--two-d-prior', 'ensemble_precision']) == 0
    study = json.loads((output / 'study.json').read_text())
    assert study['analysis']['method'] == 'enif' and study['analysis']['pet_version']
    history = json.loads((output / 'history.json').read_text())
    assert [entry['gate']['open'] for entry in history] == [False, False, True]
    assert history[1]['posterior']['mean_prediction_1d_misfit'] < history[1]['prior']['mean_prediction_1d_misfit']
    problem = chain_laplace.load_problem(output / 'laplace-problem-002.npz')
    assert problem['prior_type'] == 'ensemble_precision'
    arguments = loop.driver_arguments({**study, 'ensemble_size': 40}, 4)
    assert arguments[arguments.index('--analysis') + 1] == 'enif'
    for flags in (['--lateral-update', 'ridge'], ['--cascade-threshold', '5', '--likelihood-weight', '.5']):
        with pytest.raises(SystemExit):
            driver.parse_args(['--output-dir', str(tmp_path / 'x'), '--analysis', 'enif', *flags])
