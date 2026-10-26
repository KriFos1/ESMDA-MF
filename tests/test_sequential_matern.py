import json
import pickle

import h5py
import numpy as np
import pandas as pd
import pytest

from matern_prior import MaternPrior
import run_DA_sequential as sequential
import sequential_matern_1d as driver


def small_prior():
    return MaternPrior((6, 8), (5., 5.), [[20., 25.], [30., 15.]], [1., .5], [.8, .5])


def test_conditional_fill_keeps_kept_cells_and_reproduces_unconditional_draws():
    prior = small_prior()
    entered = np.zeros((6, 8), dtype=bool)
    entered[-2:] = True
    draws = prior.sample(4, 11)
    # Kept cells equal to an unconditional draw: the kriging correction vanishes.
    np.testing.assert_allclose(prior.condition_entered(draws, entered.ravel(), 11), draws, atol=1e-10)
    other = prior.sample(4, 12)
    filled = prior.condition_entered(other, entered.ravel(), 11)
    kept = np.r_[~entered.ravel(), ~entered.ravel()]
    np.testing.assert_array_equal(filled[kept], other[kept])
    assert not np.allclose(filled[~kept], other[~kept])
    np.testing.assert_array_equal(prior.condition_entered(other, np.zeros(48, bool), 1), other)
    np.testing.assert_allclose(prior.condition_entered(other, np.ones(48, bool), 11), draws)


def test_conditional_fill_matches_dense_conditional_moments():
    prior = small_prior()
    block = prior.blocks[0]
    n = block.n
    covariance = block.color(block.color(np.eye(n), transpose=True))
    entered = np.zeros((6, 8), dtype=bool)
    entered[:, -3:] = True
    entered = entered.ravel()
    kept = ~entered
    gain = covariance[np.ix_(entered, kept)] @ np.linalg.inv(covariance[np.ix_(kept, kept)])
    # Same draws, different kept values: the output difference is the conditional-mean difference.
    first = np.random.default_rng(1).normal(size=(n, 3))
    second = np.random.default_rng(2).normal(size=(n, 3))
    draws = np.random.default_rng(3).normal(size=(n, 3))
    difference = (block.condition_entered(first, draws, entered)
                  - block.condition_entered(second, draws, entered))[entered]
    np.testing.assert_allclose(difference, gain @ (first - second)[kept], atol=1e-8)
    # Conditional covariance: many conditional draws for one fixed kept state.
    fixed = np.repeat(first[:, :1], 6000, axis=1)
    conditional = prior.condition_entered(np.vstack([fixed, fixed]), entered, 7)[:n][entered]
    expected = (covariance[np.ix_(entered, entered)] - gain @ covariance[np.ix_(kept, entered)])
    empirical = np.cov(conditional)
    assert np.linalg.norm(empirical - expected) / np.linalg.norm(expected) < .1
    assert np.linalg.norm(conditional.mean(axis=1) - (block.mean[entered] + gain @ (first[kept, 0]
                                                                                   - block.mean[kept]))) \
        < .1 * np.sqrt(np.trace(expected))


def test_entered_cells_and_reindexing_follow_the_window_move():
    nx, nz = 5, 4
    assert driver.entered_cells(nx, nz, (0, 0)).sum() == 0
    forward = driver.entered_cells(nx, nz, (2, 0))
    assert forward[3:].all() and not forward[:3].any()
    down = driver.entered_cells(nx, nz, (-1, 1))
    assert down[0].all() and down[:, -1].all() and down.sum() == nz + nx - 1
    assert driver.entered_cells(nx, nz, (9, 0)).all()
    state = np.arange(2 * nx * nz, dtype=float)[:, None] * np.ones((1, 2))
    shifted = driver.reindex_window(state, nx, nz, (1, 0))
    fields = shifted.reshape(2, nx, nz, 2)
    np.testing.assert_array_equal(fields[0, 0, :, 0], np.arange(nz) + nz)   # column 1 moved to 0
    np.testing.assert_array_equal(fields[0, -1], fields[0, -2])               # edge value duplicated


@pytest.fixture
def driver_environment(tmp_path, monkeypatch):
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    depths_m = np.array([100., 101.524, 103.048, 104.572])
    nx, nz = 5, 6
    with h5py.File(benchmark / 'globalmodel.h5', 'w') as handle:
        handle['wellpath/Z'] = depths_m[:, None]
        handle['wellpath/X'] = (np.arange(4) * 1.524)[:, None]
        handle['wellpath/Y'] = np.zeros((4, 1))
        handle['formation/x'] = (np.arange(-10, 30) * 1.524)[:, None]
        handle['formation/z'] = (np.arange(60, 140) * 1.524)[:, None]
        handle['formation/Rh'] = np.full((80, 3, 40), 1.5)
    tool = ('6kHz', '83ft')
    depths = depths_m * sequential.meter_to_feet
    pd.DataFrame({tool: [np.full(8, 2.)] * 4}, index=depths).to_pickle(tmp_path / 'data.pkl')
    pd.DataFrame({tool: [['ABS', [.25] * 8]] * 4}, index=depths).to_pickle(tmp_path / 'var.pkl')
    manifest = {'benchmark': str(benchmark), 'dims': [nx, 1, nz], 'spacing_ft': [5., 5., 5.],
                'data': str(tmp_path / 'data.pkl'), 'variance': str(tmp_path / 'var.pkl'),
                'prior_background': np.log10(1.5), 'prior_reservoir': np.log10(30.),
                'prior_mode_std': .1, 'prior_reservoir_fraction': .15, 'prior_ratio_max': 3.,
                'prior_latent_correlation_x_ft': 80., 'prior_latent_correlation_z_ft': 35.,
                'dx_ft': 5., 'dz_ft': 5., 'rv_parameterization': {'latent_std': .12}}
    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'run.json').write_text(json.dumps(manifest))

    def setup(benchmark_dir, tools):
        monkeypatch.setattr(sequential, 'TVD', depths, raising=False)
        monkeypatch.setattr(sequential, 'MD', depths, raising=False)
        monkeypatch.setattr(sequential, 'WX', np.arange(4) * 5., raising=False)
        monkeypatch.setattr(sequential, 'WY', np.zeros(4), raising=False)
        monkeypatch.setattr(sequential, 'wellpath', {}, raising=False)

    def fake_pred(param, index, simfidelity='1D'):
        assert simfidelity == '1D' and len(param) == 2 * nz
        return np.full(8, float(np.mean(param[:nz])))

    monkeypatch.setattr(sequential, 'setup_simulators', setup)
    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    for name in ('dims', 'dX', 'dY', 'dZ', 'tools', 'max_ratio', 'log_rv_cap', 'anisotropy',
                 'current_shift'):   # the driver sets these; restore them for later tests
        monkeypatch.setattr(sequential, name, getattr(sequential, name, None), raising=False)
    return tmp_path, reference


def test_driver_chains_posteriors_through_window_moves(tmp_path, driver_environment):
    _, reference = driver_environment
    output = tmp_path / 'run'
    assert driver.main(['--reference', str(reference), '--output-dir', str(output),
                        '--stop', '3', '--ensemble-size', '12', '--mda-steps', '3']) == 0
    history = json.loads((output / 'history.json').read_text())
    assert [entry['logging_index'] for entry in history] == [0, 1, 2]
    assert history[0]['entered_cells'] == 30 and history[1]['entered_cells'] < 30
    assert history[1]['window_movement'] == [1, 1]
    for entry in history:
        assert len(entry['steps']) == 4 and entry['posterior']['matern_2p_over_n_rh_median'] > 0
        assert entry['posterior']['mean_prediction_1d_misfit'] < entry['prior']['mean_prediction_1d_misfit']
    with open(output / 'inversion_results_assim_1.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    assert saved['selected_fidelity'] == '1D' and saved['anisotropy'] == 'log_clipped'
    assert saved['prior_style'] == 'matern' and saved['misfit_threshold'] is None
    assert np.shape(saved['posterior_ensemble_params']) == (12, 60)
    assert np.isclose(np.reciprocal(saved['covariance_inflation']).sum(), 1.)
    with open(output / 'inversion_results_assim_0.pkl', 'rb') as stream:
        first = pickle.load(stream)
    # Kept cells of index 1's prior are index 0's posterior, moved by one cell in x and z.
    carried = driver.reindex_window(first['post_mda_params'], 5, 6, (1, 1))
    kept = ~np.r_[driver.entered_cells(5, 6, (1, 1)).ravel(), driver.entered_cells(5, 6, (1, 1)).ravel()]
    np.testing.assert_array_equal(saved['prior_params'][kept], carried[kept])


def test_relaxation_keeps_the_prior_invariant_and_interpolates_its_limits():
    prior = small_prior()
    state = prior.sample(300, 5)
    rng = np.random.default_rng(9)
    np.testing.assert_array_equal(driver.relax_to_prior(prior, state, 1., rng), state)
    fully = driver.relax_to_prior(prior, state, 0., np.random.default_rng(9))

    def score(models):   # 2P/n over both blocks: 1 for prior draws
        return float(np.mean(prior.penalty(models) / prior.cells))
    assert not np.allclose(fully, state) and abs(score(fully) - 1) < .15
    relaxed = state
    for _ in range(20):
        relaxed = driver.relax_to_prior(prior, relaxed, .9, rng)
    assert abs(score(relaxed) - 1) < .15
    np.testing.assert_allclose(relaxed.mean(axis=1), prior.mean, atol=.25)
    collapsed = np.repeat(prior.mean[:, None] + 3., 300, axis=1)   # no spread, far from the mean
    pulled = driver.relax_to_prior(prior, collapsed, .9, rng)
    offset = (pulled - prior.mean[:, None]).mean(axis=1)
    np.testing.assert_allclose(offset, 2.7, atol=.15)
    assert .3 < pulled.std(axis=1).mean() / prior.std.mean() < .6      # sqrt(1 - .81) = .44
    with pytest.raises(ValueError):
        driver.relax_to_prior(prior, state, 1.5, rng)


def test_driver_applies_the_memory_relaxation_between_points(tmp_path, driver_environment):
    _, reference = driver_environment
    output = tmp_path / 'relaxed'
    driver.main(['--reference', str(reference), '--output-dir', str(output), '--stop', '3',
                 '--ensemble-size', '12', '--mda-steps', '2', '--memory-ft', '10'])
    history = json.loads((output / 'history.json').read_text())
    assert history[0]['relaxation_rho'] == 1. and history[0]['along_well_step_ft'] is None
    assert np.isclose(history[1]['relaxation_rho'], np.exp(-5. / 10.), atol=1e-3)
    with open(output / 'inversion_results_assim_1.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    assert saved['memory_ft'] == 10.
    with open(output / 'inversion_results_assim_0.pkl', 'rb') as stream:
        first = pickle.load(stream)
    carried = driver.reindex_window(first['post_mda_params'], 5, 6, (1, 1))
    kept = ~np.r_[driver.entered_cells(5, 6, (1, 1)).ravel(), driver.entered_cells(5, 6, (1, 1)).ravel()]
    assert not np.allclose(saved['prior_params'][kept], carried[kept])   # relaxed, not copied
    study = json.loads((output / 'study.json').read_text())
    assert study['relaxation']['memory_ft'] == 10.


def test_driver_resumes_from_the_last_checkpoint(tmp_path, driver_environment):
    _, reference = driver_environment
    output = tmp_path / 'resumable'
    common = ['--reference', str(reference), '--output-dir', str(output), '--ensemble-size', '12',
              '--mda-steps', '2']
    assert driver.main([*common, '--stop', '2']) == 0
    assert driver.main([*common, '--stop', '4', '--resume']) == 0
    history = json.loads((output / 'history.json').read_text())
    assert [entry['logging_index'] for entry in history] == [0, 1, 2, 3]
    assert [entry['previous_index'] for entry in history] == [None, 0, 1, 2]
    study = json.loads((output / 'study.json').read_text())
    assert study['stop'] == 4 and len(study['resumes']) == 1
    resume = study['resumes'][0]
    assert {key: resume[key] for key in ('from_index', 'stop', 'memory_ft', 'gate_threshold', 'on_gate')} == {
        'from_index': 1, 'stop': 4, 'memory_ft': None, 'gate_threshold': None, 'on_gate': 'stop'}
    assert resume['lateral_update']['mode'] == 'full' and resume['cascade_threshold'] is None
    with open(output / 'inversion_results_assim_1.pkl', 'rb') as stream:
        before = pickle.load(stream)
    with open(output / 'inversion_results_assim_2.pkl', 'rb') as stream:
        after = pickle.load(stream)
    kept = ~np.r_[driver.entered_cells(5, 6, (1, 1)).ravel(), driver.entered_cells(5, 6, (1, 1)).ravel()]
    np.testing.assert_array_equal(after['prior_params'][kept],
                                  driver.reindex_window(before['post_mda_params'], 5, 6, (1, 1))[kept])
    with pytest.raises(ValueError):
        driver.main([*common, '--stop', '4', '--resume'])
    # A resume must keep the chain's reference (its run.json supplies the data).
    other = tmp_path / 'other-reference'
    other.mkdir()
    (other / 'run.json').write_text((reference / 'run.json').read_text())
    with pytest.raises(ValueError, match='started from'):
        driver.main(['--reference', str(other), *common[2:], '--stop', '4', '--resume'])
