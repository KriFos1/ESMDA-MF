from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import run_DA_sequential as sequential
from esmda_core import (LOG_RH_LIMITS, LocalWindow, MemberEvaluationError,
                        clamp_log_rh, measurement_misfit, noise_floor_mask,
                        observation_vectors, optimize_member, optimize_members,
                        optimizer_jacobian, project_to_modes, shift_posterior,
                        shift_posterior_1d, update_members_once)
from esmda_remote2d import Remote2D, SimBackendConfig, UDAR_COMPONENTS
from write_data import generate_data, read_las


def test_import_and_help_preserve_existing_simulator_files(tmp_path):
    sentinel = tmp_path / 'En_0' / 'keep.txt'
    sentinel.parent.mkdir()
    sentinel.write_text('existing run')
    script = Path(sequential.__file__)
    result = subprocess.run([sys.executable, '-B', str(script), '--help'], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--output-dir' in result.stdout
    assert sentinel.read_text() == 'existing run'
    assert list(tmp_path.iterdir()) == [sentinel.parent]


def test_window_is_small_centered_and_idempotent():
    window = LocalWindow()
    origin, movement = window.update((100., 7., 1000.))
    assert window.dims == (15, 5, 20)
    assert np.prod(window.dims[::2]) == 300
    assert movement == (0, 0)
    assert origin['x'] + 7.5 * 5 == 100
    assert origin['z'] + 10.5 * 5 == 1000
    assert origin['y'] + 2.5 * 5 == 7
    shifted, movement = window.update((111., 9., 994.))
    assert movement == (2, -1)
    assert shifted['x'] - origin['x'] == 10
    assert shifted['z'] - origin['z'] == -5
    repeated, movement = window.update((111., 9., 994.))
    assert repeated == shifted
    assert movement == (0, 0)


def test_posterior_shift_tracks_cells_and_extends_edges():
    fields = np.zeros((1, 2, 3, 4))
    fields[0, 0, 2, 1] = 7
    fields[0, 1, 1, 0] = 9
    result = shift_posterior(fields.reshape(1, -1), (3, 5, 4), (1, -1)).reshape(fields.shape)
    assert result[0, 0, 1, 2] == 7
    assert result[0, 0, 2, 2] == 7
    assert result[0, 1, 0, 0] == result[0, 1, 0, 1] == 9
    assert fields[0, 0, 2, 1] == 7


def test_worker_jacobian_cell_and_component_contract(tmp_path):
    from pxwf.execution import worker
    adapter = Remote2D(SimBackendConfig(), (3, 2, 4), 5., 5., 5., [('6kHz', '83ft')])
    metadata = {'tools': [['6kHz', '83ft']], 'jacobian': True,
                'full_jacobian': True, 'y_layers': 2, 'bit_column': 1}
    output = tmp_path / 'outputs'
    (output / 'jacobian').mkdir(parents=True)
    (output / 'UDAR_83ft_6kHz.las').write_text('% MD_ft ' + ' '.join(UDAR_COMPONENTS) + '\n0 1 2 3 4 5 6 7 8\n')
    jac = np.zeros((2, 8, 4, 2, 3))
    jac[0, 6, 1, :, 2] = [2., 3.]
    jac[1, 7, 3, :, 0] = [7., 11.]
    with h5py.File(output / 'jacobian/jac_logpt1_tool1.h5', 'w') as handle:
        handle['jac'] = jac
    prediction, _, full = worker.read_outputs(tmp_path, metadata, 4, 3)
    horizontal, vertical = adapter._adapt_full_jacobian(full)
    np.testing.assert_array_equal(adapter._prediction(prediction), np.arange(1, 9))
    assert horizontal.shape == vertical.shape == (8, 12)
    assert horizontal[6, 2 * 4 + 1] == 5
    assert vertical[7, 0 * 4 + 3] == 18
    assert np.count_nonzero(horizontal) == np.count_nonzero(vertical) == 1


def test_field_layout_and_log_chain_rule_against_finite_difference(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (3, 2, 4))
    parameters = np.r_[np.linspace(.4, 1.4, 12), np.linspace(-.7, .8, 12)]
    fields, _, latent = sequential._build_param_dict(parameters)
    adapter = Remote2D(SimBackendConfig(), (3, 2, 4), 5., 5., 5., [('6kHz', '83ft')])
    tile = adapter.to_tile(fields)
    assert tile[0, 3, 2] == parameters[11]
    assert tile[0, 1, 2] == parameters[9]
    weights = np.arange(1, 25).reshape(2, 12) / 24

    def response(values):
        fields, _, _ = sequential._build_param_dict(values)
        rh = fields['rh'].reshape(3, 2, 4, order='F')[:, 0].ravel()
        rv = fields['rv'].reshape(3, 2, 4, order='F')[:, 0].ravel()
        return weights[0] @ np.exp(rh) + weights[1] @ np.exp(rv)

    log_rh = parameters[:12]
    log_rv = log_rh + np.log(5) * norm.cdf(parameters[12:])
    jac = optimizer_jacobian((weights[0] * np.exp(log_rh))[None],
                             (weights[1] * np.exp(log_rv))[None], latent)
    direction = np.random.default_rng(17).normal(size=24)
    step = 1e-5
    fd = (response(parameters + step * direction) - response(parameters - step * direction)) / (2 * step)
    np.testing.assert_allclose(jac @ direction, [fd], rtol=1e-7)


def test_metadata_uses_current_orientation_and_origin():
    adapter = Remote2D(SimBackendConfig(), (3, 5, 4), 5., 5., 5., [('6kHz', '83ft')])
    well = {key: np.array([[a], [b]]) for key, a, b in
            [('X', 1, 2), ('Y', 3, 4), ('Z', 100, 101), ('Distance', 5, 6),
             ('Inclination', 80, 85), ('Azimuth', 0, 15)]}
    meta = adapter.build_metadata(1, {'x': 10, 'y': 20, 'z': 30}, well)
    assert meta['well']['Inclination'] == 85
    assert meta['well']['Azimuth'] == 15
    np.testing.assert_allclose(meta['origin'], np.array([10, 20, 30]) / 3.28084)
    assert meta['solver_tolerance'] == 1e-6


def test_local_backend_honors_cpu_and_worker_requests(tmp_path):
    adapter = Remote2D(SimBackendConfig(cpus=3, concurrency=2), (3, 5, 4),
                       5., 5., 5., [('6kHz', '83ft')], jobs_dir=str(tmp_path / 'jobs'))
    backend = adapter.backend()
    assert backend.resources.cpus_per_rank == 3
    assert backend.workers == 2
    assert not (tmp_path / 'jobs').exists()


def test_measurement_misfit_is_reduced_chi_square():
    prediction = np.array([1., 2., 3.])
    observations = np.array([0., 2., 4.])
    covariance = np.array([1., 4., 4.])
    assert measurement_misfit(prediction, observations, covariance) == pytest.approx(1.25 / 3)
    with pytest.raises(ValueError):
        measurement_misfit(prediction, observations, np.zeros(3))
    with pytest.raises(ValueError):
        measurement_misfit(np.ones(2), observations, covariance)


def test_variance_selection_handles_rowwise_and_legacy_envelopes():
    tool = ('6kHz', '83ft')
    data = pd.DataFrame({tool: [np.ones(8), np.ones(8) * 2]})
    rowwise = pd.DataFrame({tool: [['ABS', [1.] * 8], ['ABS', [4.] * 8]]})
    envelope = ['ABS', [1.] * 8, [4.] * 8]
    legacy = pd.DataFrame({tool: [envelope, envelope]})
    for variance in (rowwise, legacy):
        observations, covariance = observation_vectors(data, variance, 1, [tool])
        np.testing.assert_array_equal(observations, np.full(8, 2))
        np.testing.assert_array_equal(covariance, np.full(8, 4))


def test_las_header_order_and_variance_generation(tmp_path):
    benchmark = tmp_path / 'benchmark'
    logs = benchmark / 'outputs_2D'
    logs.mkdir(parents=True)
    with h5py.File(benchmark / 'globalmodel.h5', 'w') as handle:
        handle['wellpath/Z'] = [[10.], [11.]]
        handle['wellpath/Distance'] = [[10.], [11.]]
    names = list(reversed(UDAR_COMPONENTS))
    path = logs / 'UDAR3D_83ft_6kHz.las'
    path.write_text('% MD_ft ' + ' '.join(names) + '\n32.8084 8 7 6 5 4 3 2 1\n36.08924 16 14 12 10 8 6 4 2\n')
    _, values = read_las(path)
    np.testing.assert_array_equal(values[0], np.arange(1, 9))
    data, variances = generate_data(benchmark, [('6kHz', '83ft')], tmp_path / 'data')
    _, first = observation_vectors(data, variances, 0, list(data.columns))
    _, second = observation_vectors(data, variances, 1, list(data.columns))
    np.testing.assert_allclose(second, first * 4)
    assert len(variances.iloc[1, 0]) == 2


def test_conditioning_respects_actual_columns(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (4, 5, 3))
    columns = [np.full((6, 2), .5), np.full((6, 2), 1.5)]
    previous = np.zeros((2, 24))
    ensemble, covariance, _ = sequential.resample_prior_multiple_1D_to_2D(2, columns, [1, 3], previous)
    fields = ensemble.reshape(2, 4, 3, 2)
    np.testing.assert_array_equal(fields[:, 1], .5)
    np.testing.assert_array_equal(fields[:, 3], 1.5)
    assert covariance.shape == (24, 24)


def test_lm_updates_prediction_and_preserves_accepted_result_after_failure():
    evaluated = []

    def evaluate(parameters):
        evaluated.append(parameters.copy())
        if len(evaluated) == 3:
            raise RuntimeError('failed trial')
        return parameters.copy(), np.eye(1)

    def update(parameters, prediction, *args):
        np.testing.assert_array_equal(parameters, prediction)
        return np.array([-1.])

    def loss(prediction, observation, *args):
        return float(np.sum((prediction - observation) ** 2))

    result = optimize_member(np.array([2.]), np.array([0.]), np.zeros(1), np.eye(1),
                             np.ones(1), evaluate, update, loss, max_iterations=2, max_trials=1)
    np.testing.assert_array_equal(result['parameters'], [1.])
    np.testing.assert_array_equal(result['prediction'], [1.])
    assert result['loss'] == 1
    assert result['history'] == [4, 1, 1]
    assert result['trial_errors'] == ['RuntimeError: failed trial']


def test_clamp_log_rh_bounds_horizontal_block_only():
    vector = np.array([-3., 6., 0.5, 2.])
    np.testing.assert_allclose(clamp_log_rh(vector), [-1., 4., 0.5, 2.])
    np.testing.assert_allclose(clamp_log_rh(np.array([2.5])), [2.5])
    members = np.array([[-3., 6.], [0.5, 2.]])
    clamped = clamp_log_rh(members)
    np.testing.assert_allclose(clamped, [[-1., 4.], [0.5, 2.]])
    assert members[0, 0] == -3.
    assert LOG_RH_LIMITS == (-1., 4.)


def test_noise_floor_mask_flags_only_floor_variance_channels():
    observations = np.array([0., 5., 1e-4])
    covariance = np.array([1e-6, 4., 1e-8])
    np.testing.assert_array_equal(noise_floor_mask(observations, covariance),
                                  [False, False, True])
    with pytest.raises(ValueError):
        noise_floor_mask(observations, np.zeros(3))
    with pytest.raises(ValueError):
        noise_floor_mask(observations[:2], covariance)


def test_initial_evaluation_retries_once_then_raises_member_error():
    calls = []

    def always_fails(parameters):
        calls.append(1)
        raise RuntimeError('solver crashed')

    def update(*args):
        return np.zeros(1)

    def loss(*args):
        return 1.

    with pytest.raises(MemberEvaluationError):
        optimize_member(np.array([2.]), np.array([0.]), np.zeros(1), np.eye(1),
                        np.ones(1), always_fails, update, loss, max_iterations=1)
    assert len(calls) == 2

    flaky = {'calls': 0}

    def transient(parameters):
        flaky['calls'] += 1
        if flaky['calls'] == 1:
            raise RuntimeError('transient')
        return np.array([1.]), np.eye(1)

    result = optimize_member(np.array([2.]), np.array([0.]), np.zeros(1), np.eye(1),
                             np.ones(1), transient, update, loss, max_iterations=0)
    assert result['loss'] == 1.
    assert flaky['calls'] == 2


def test_controller_substitutes_member_when_initial_2d_evaluation_fails(
        tmp_path, monkeypatch, controller_environment):
    base, benchmark = controller_environment
    state = {'grad_calls': 0}

    def fake_pred(param, index, simfidelity='2D'):
        return np.ones(8) * 5

    def fake_grad_multi(param_list, index):
        results = []
        for param in param_list:
            state['grad_calls'] += 1
            if state['grad_calls'] in (2, 3):
                results.append(RuntimeError('segfault'))
            else:
                results.append((np.ones(8), np.zeros((8, len(param)))))
        return results

    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    monkeypatch.setattr(sequential, 'simulate_and_grad_multi', fake_grad_multi)
    output = tmp_path / 'run'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--nx', '3', '--nz', '4', '--ensemble-size', '6', '--opt-members', '2',
                     '--max-iterations', '0'])
    for index in range(3):
        result = pd.read_pickle(output / f'inversion_results_assim_{index}.pkl')
        assert result['selected_fidelity'] == '2D'
        assert np.shape(result['posterior_params']) == (2, 24)
        if index == 0:
            assert result['substituted_members'] == 1
            failure = result['solver_failures'][0]
            assert failure['slot'] == 1
            assert failure['member'] != failure['substitute']
            assert result['member_candidates'][1] == failure['substitute']
            assert result['two_d_calls'] == 4
        else:
            assert result['substituted_members'] == 0
            assert result['two_d_calls'] == 2


def test_controller_abandons_2d_after_substitution_cap_and_falls_back_to_1d(
        tmp_path, monkeypatch, controller_environment):
    base, benchmark = controller_environment

    def fake_pred(param, index, simfidelity='2D'):
        return np.ones(8) * 5

    def fake_grad_multi(param_list, index):
        return [RuntimeError('segfault')] * len(param_list)

    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    monkeypatch.setattr(sequential, 'simulate_and_grad_multi', fake_grad_multi)
    output = tmp_path / 'run'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--nx', '3', '--nz', '4', '--ensemble-size', '6', '--opt-members', '2',
                     '--max-iterations', '0', '--max-substitutions', '3'])
    for index in range(3):
        result = pd.read_pickle(output / f'inversion_results_assim_{index}.pkl')
        assert result['selected_fidelity'] == '1D'
        assert result['two_d_mode'] == 'abandoned'
        assert result['substituted_members'] == 3
        assert result['escalation_abandoned'].startswith('Initial evaluation failed twice')
        assert len(result['solver_failures']) == 3
        assert np.shape(result['posterior_params']) == (2, 8)
        assert result['two_d_calls'] > 0
        assert result['posterior_prediction_kind'] == 'forward'

def test_controller_masks_structural_zero_channels(tmp_path, monkeypatch, controller_environment):
    base, benchmark = controller_environment
    tool = ('6kHz', '83ft')
    depths = np.array([100., 101.524, 103.048]) * sequential.meter_to_feet
    observations = np.array([1., 1., 0., 0., 1., 1., 1., 1.])
    variance = np.array([1., 1., 1e-8, 1e-8, 1., 1., 1., 1.])
    pd.DataFrame({tool: [observations] * 3}, index=depths).to_pickle(base / 'data.pkl')
    pd.DataFrame({tool: [['ABS', variance.tolist()]] * 3}, index=depths).to_pickle(base / 'var.pkl')

    def fake_pred(param, index, simfidelity='2D'):
        return np.array([1., 5., 1., 5., 1., 5., 1., 5.])

    def fake_grad_multi(param_list, index):
        prediction = np.array([1., 5., 1., 5., 1., 5., 1., 5.])
        return [(prediction, np.zeros((8, len(param)))) for param in param_list]

    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    monkeypatch.setattr(sequential, 'simulate_and_grad_multi', fake_grad_multi)
    output = tmp_path / 'run'
    sequential.main(['--benchmark', str(benchmark), '--data', str(base / 'data.pkl'),
                     '--variance', str(base / 'var.pkl'), '--output-dir', str(output),
                     '--nx', '3', '--nz', '4', '--ensemble-size', '4', '--opt-members', '2',
                     '--max-iterations', '1'])
    result = pd.read_pickle(output / 'inversion_results_assim_0.pkl')
    assert result['masked_channels'] == 4
    assert result['masked_components'] == ('USDP', 'USDA')
    assert result['observation_mask'].sum() == 4
    assert result['observations'].shape == (4,)
    assert result['data_covariance'].shape == (4,)
    assert result['posterior_predictions'][0].shape == (4,)
    kept = np.where(result['observation_mask'])[0]
    np.testing.assert_allclose(kept, [4, 5, 6, 7])
    np.testing.assert_allclose(result['observations'], observations[kept])
    np.testing.assert_allclose(result['data_covariance'],
                               variance[kept] * result['final_inflation_factor'])
    assert result['misfit']['0D_initial'] == pytest.approx(32. / 4.)


def test_optimize_members_batches_members_and_stops_early():
    batches = []

    def evaluate_multi(params_list):
        batches.append([float(p[0]) for p in params_list])
        return [(np.array([float(parameters[0])]), np.eye(1)) for parameters in params_list]

    def update(parameters, prediction, *args):
        return np.array([-1.8 if float(parameters[0]) > 1. else -0.01])

    def loss(prediction, observation, *args):
        return float(np.sum((prediction - observation) ** 2))

    records = optimize_members([np.array([4.]), np.array([0.5])],
                               [np.array([0.]), np.array([0.])],
                               np.zeros(1), np.eye(1), np.ones(1),
                               evaluate_multi, update, loss, max_iterations=10,
                               max_rejects=2, improve_tol=0.05)
    fast, slow = records
    assert fast['loss'] < 0.2
    assert slow['loss'] == pytest.approx(0.48 ** 2)
    assert len(slow['history']) == 3
    assert len(fast['history']) > len(slow['history'])
    assert len(batches[0]) == 2
    assert any(len(batch) == 1 for batch in batches)
    assert all(len(batch) <= 2 for batch in batches)


def test_optimize_members_rejects_failed_fields_and_damps():
    calls = {'n': 0}

    def evaluate_multi(params_list):
        calls['n'] += 1
        results = []
        for parameters in params_list:
            if calls['n'] == 2 and float(parameters[0]) > 2:
                results.append(RuntimeError('solver convergence error'))
            else:
                results.append((np.array([float(parameters[0])]), np.eye(1)))
        return results

    def update(parameters, prediction, *args):
        return np.array([-0.5])

    def loss(prediction, observation, *args):
        return float(np.sum((prediction - observation) ** 2))

    records = optimize_members([np.array([2.]), np.array([4.])],
                               [np.array([0.]), np.array([0.])],
                               np.zeros(1), np.eye(1), np.ones(1),
                               evaluate_multi, update, loss, max_iterations=5,
                               max_rejects=2)
    failures = [record for record in records if record['trial_errors']]
    assert len(failures) == 1
    assert 'RuntimeError: solver convergence error' in failures[0]['trial_errors'][0]


def test_optimize_members_substitutes_failed_initial_member():
    batches = []

    def evaluate_multi(params_list):
        batches.append([float(p[0]) for p in params_list])
        results = []
        for parameters in params_list:
            if float(parameters[0]) == 7.:
                results.append(RuntimeError('segfault'))
            else:
                results.append((np.array([float(parameters[0])]), np.eye(1)))
        return results

    def update(parameters, prediction, *args):
        return np.zeros(1)

    def loss(prediction, observation, *args):
        return float(np.sum((prediction - observation) ** 2))

    substitutes = []

    def substitute(slot, exc):
        substitutes.append((slot, str(exc)))
        return np.array([3.]), np.array([0.])

    records = optimize_members([np.array([1.]), np.array([7.])],
                               [np.array([0.]), np.array([0.])],
                               np.zeros(1), np.eye(1), np.ones(1),
                               evaluate_multi, update, loss, max_iterations=1,
                               substitute=substitute)
    assert substitutes == [(1, 'segfault')]
    assert batches[0] == [1., 7.]
    assert 7. in batches[1]
    assert len(records) == 2
    assert records[1]['loss'] == pytest.approx(9.)

    with pytest.raises(MemberEvaluationError):
        optimize_members([np.array([7.])], [np.array([0.])], np.zeros(1), np.eye(1), np.ones(1),
                         evaluate_multi, update, loss, max_iterations=1, substitute=None)


def test_prior_spread_scales_residual_and_covariance(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (4, 5, 3))
    columns = [np.full((6, 2), .5), np.full((6, 2), 1.5)]
    previous = np.zeros((2, 24))
    wide, cov_wide, _ = sequential.resample_prior_multiple_1D_to_2D(2, columns, [1, 3], previous)
    tight, cov_tight, _ = sequential.resample_prior_multiple_1D_to_2D(
        2, columns, [1, 3], previous, posterior_spread=(1e-6, 1e-6))
    fields_wide = wide.reshape(2, 4, 3, 2)
    fields_tight = tight.reshape(2, 4, 3, 2)
    np.testing.assert_allclose(fields_wide[:, 1], .5)
    np.testing.assert_allclose(fields_tight[:, 1], .5)
    spread_wide = fields_wide[:, 2].std(axis=-1).mean()
    spread_tight = fields_tight[:, 2].std(axis=-1).mean()
    assert 0. < spread_tight < spread_wide
    assert np.trace(cov_tight) < np.trace(cov_wide)


def test_run_grad_multi_maps_worker_failures_to_none_jacobians(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (3, 5, 4))
    adapter = Remote2D(SimBackendConfig(), (3, 5, 4), 5., 5., 5., [('6kHz', '83ft')])
    parameters = [sequential._build_param_dict(np.full(24, value))[0]
                  for value in (0.5, 1.0, 1.5)]

    class Backend:
        def trusts_ad(self, fidelity):
            return True

        def evaluate_batch(self, metadata, fields):
            assert metadata['tolerate_failures'] is True
            good, jac = np.ones(8), np.ones((8, 24))
            empty = np.empty((0, 0))
            if fields.shape[0] == 1:
                return np.stack([good]), [empty], [jac]
            nan_row = np.full(8, np.nan)
            predictions = np.stack([good, nan_row, good])
            return predictions, [empty] * 3, [jac, empty, jac]

    adapter._backend = Backend()
    results = adapter.run_grad_multi(parameters[:1], 0, {'x': 0., 'y': 0., 'z': 0.},
                                     {'X': np.array([[1.]]), 'Y': np.array([[0.]]),
                                      'Z': np.array([[100.]]), 'Distance': np.array([[5.]]),
                                      'Inclination': np.array([[85.]]), 'Azimuth': np.array([[0.]])})
    assert len(results) == 1 and results[0][1] is not None
    results = adapter.run_grad_multi(parameters, 0, {'x': 0., 'y': 0., 'z': 0.},
                                     {'X': np.array([[1.]]), 'Y': np.array([[0.]]),
                                      'Z': np.array([[100.]]), 'Distance': np.array([[5.]]),
                                      'Inclination': np.array([[85.]]), 'Azimuth': np.array([[0.]])})
    assert len(results) == 3
    assert results[0][1] is not None and results[0][1].shape == (8, 12)
    assert results[1][1] is None and results[1][2] is None
    assert results[2][1] is not None


def test_run_grad_multi_degrades_misaligned_collect_without_raising(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (3, 5, 4))
    adapter = Remote2D(SimBackendConfig(), (3, 5, 4), 5., 5., 5., [('6kHz', '83ft')])
    parameters = [sequential._build_param_dict(np.full(24, value))[0]
                  for value in range(5)]

    class Backend:
        def trusts_ad(self, fidelity):
            return True

        def evaluate_batch(self, metadata, fields):
            good, jac = np.ones(8), np.ones((8, 24))
            empty = np.full(8, np.nan)
            # Reproduces the historical collect() bug: field 0 failed first,
            # so the full-Jacobian list lost its leading placeholder.
            predictions = np.stack([empty, good, good, empty, empty])
            return predictions, [empty] * 5, [jac, jac, empty, empty]

    adapter._backend = Backend()
    results = adapter.run_grad_multi(parameters, 0, {'x': 0., 'y': 0., 'z': 0.},
                                     {'X': np.array([[1.]]), 'Y': np.array([[0.]]),
                                      'Z': np.array([[100.]]), 'Distance': np.array([[5.]]),
                                      'Inclination': np.array([[85.]]), 'Azimuth': np.array([[0.]])})
    assert len(results) == 5
    assert all(j_rh is None for _, j_rh, _ in results)

    class NoFullBackend(Backend):
        def evaluate_batch(self, metadata, fields):
            return np.ones((fields.shape[0], 8)), [np.empty((0, 0))] * fields.shape[0]

    adapter._backend = NoFullBackend()
    with pytest.raises(RuntimeError):
        adapter.run_grad_multi(parameters, 0, {'x': 0., 'y': 0., 'z': 0.},
                               {'X': np.array([[1.]]), 'Y': np.array([[0.]]),
                                'Z': np.array([[100.]]), 'Distance': np.array([[5.]]),
                                'Inclination': np.array([[85.]]), 'Azimuth': np.array([[0.]])})


def test_update_members_once_takes_one_batched_linearized_step():
    batches = []

    def evaluate_multi(params_list):
        batches.append([float(p[0]) for p in params_list])
        return [(np.array([float(p[0])]), np.array([[2.]])) for p in params_list]

    def update(parameters, prediction, *args):
        return np.array([-0.4])

    def loss(prediction, observation, *args):
        return float(np.sum((prediction - observation) ** 2))

    records = update_members_once([np.array([4.]), np.array([1.])],
                                  [np.array([0.]), np.array([0.])],
                                  np.zeros(1), np.eye(1), np.ones(1),
                                  evaluate_multi, update, loss)
    assert len(batches) == 1 and len(batches[0]) == 2
    first, second = records
    np.testing.assert_allclose(first['parameters'], [3.6])
    np.testing.assert_allclose(first['prediction'], [4.0 + 2.0 * -0.4])
    assert first['loss'] == pytest.approx((3.2 - 0.) ** 2)
    assert first['history'] == [16.0, pytest.approx(3.2 ** 2)]
    np.testing.assert_allclose(second['parameters'], [0.6])
    assert second['loss'] == pytest.approx(0.2 ** 2)

    def clamped_evaluate(params_list):
        return [(np.array([float(p[0])]), np.array([[2., 0.]])) for p in params_list]

    clamped = update_members_once([np.array([4., 0.5])], [np.array([0.])], np.zeros(2),
                                  np.eye(2), np.ones(1), clamped_evaluate,
                                  lambda *args: np.array([-20., 0.]), loss)
    np.testing.assert_allclose(clamped[0]['parameters'], [-1., 0.5])
    np.testing.assert_allclose(clamped[0]['prediction'], [4.0 + 2.0 * -5.0])

    def failing(params_list):
        return [RuntimeError('segfault') if float(p[0]) == 4. else (np.array([float(p[0])]),
                                                                    np.eye(1)) for p in params_list]

    with pytest.raises(MemberEvaluationError):
        update_members_once([np.array([4.])], [np.array([0.])], np.zeros(1), np.eye(1),
                            np.ones(1), failing, update, loss)


def test_controller_single_update_mode_runs_one_2d_batch(tmp_path, monkeypatch,
                                                          controller_environment):
    base, benchmark = controller_environment
    batches = []

    def fake_pred(param, index, simfidelity='2D'):
        return np.ones(8) * 5

    def fake_grad_multi(param_list, index):
        batches.append(len(param_list))
        return [(np.ones(8), -np.ones((8, len(param)))) for param in param_list]

    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    monkeypatch.setattr(sequential, 'simulate_and_grad_multi', fake_grad_multi)
    output = tmp_path / 'run'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--nx', '3', '--nz', '4', '--ensemble-size', '4', '--opt-members', '2',
                     '--max-iterations', '5', '--single-update', '--force-2d'])
    for index in range(3):
        result = pd.read_pickle(output / f'inversion_results_assim_{index}.pkl')
        assert result['two_d_mode'] == 'single-update'
        assert result['posterior_prediction_kind'] == 'linearized'
        assert result['posterior_forward_verified'] is False
        assert result['misfit']['after_final_kind'] == 'linearized'
        assert len(result['prior_forward_misfit']) == 2
        assert result['two_d_calls'] == 2
        assert all(len(history) == 2 for history in result['optimizer_history'])
    assert batches == [2, 2, 2]


def test_prior_cli_conversion_preserves_physical_levels_and_width():
    args, _ = sequential.parse_args(['--output-dir', 'unused'])
    background, reservoir, width = sequential.prior_parameters_ln(args)
    np.testing.assert_allclose(np.exp([background, reservoir]), [1.5, 30.])
    assert np.exp(width) == pytest.approx(10 ** .10)
    radius = 2 * width
    projected = project_to_modes(np.array([-20., 20.]), background, reservoir, radius)
    np.testing.assert_allclose(np.exp(projected), [1.5 / 10 ** .2, 30 * 10 ** .2])
    for flag in ['--prior-background', '--prior-mode-std', '--prior-switch-prob']:
        with pytest.raises(SystemExit):
            sequential.parse_args(['--output-dir', 'unused', flag, 'nan'])


def test_projection_toggle_keeps_identical_chaining_draws(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (3, 5, 4))
    bg, res = np.log([1.5, 30.])
    width = np.log(10) * .1
    trend = np.r_[np.full(4, res + 2.), np.zeros(4)][:, None]
    np.random.seed(21)
    on, _, _ = sequential.sample_chained_prior_1D(10, trend, bg, res, width, .15, 0., True)
    np.random.seed(21)
    off, _, _ = sequential.sample_chained_prior_1D(10, trend, bg, res, width, .15, 0., False)
    np.testing.assert_array_equal(on[4:], off[4:])
    assert np.all(on[:4] <= res + 2 * width)
    assert np.all(off[:4] > res + 2 * width)


def test_plot_labels_old_single_update_as_linearized():
    from plot_curtain import prediction_kind
    assert prediction_kind({'two_d_mode': 'single-update'}) == 'linearized'
    assert prediction_kind({'selected_fidelity': '1D'}) == 'forward'
    assert prediction_kind({'posterior_prediction_kind': 'forward'}) == 'forward'


def test_posterior_forward_check_uses_original_covariance_and_saved_mask(tmp_path, monkeypatch):
    import hashlib
    import json
    import pickle
    import check_posterior

    checkpoint = tmp_path / 'point.pkl'
    mask = np.array([True, True, False, False, False, False, False, False])
    saved = {'selected_fidelity': '2D', 'dims': (3, 5, 4), 'spacing_ft': (5., 5., 5.),
             'tools': [('6kHz', '83ft')], 'logging_index': 0,
             'shift_ft': {'x': 0., 'y': 0., 'z': 0.}, 'posterior_params': [np.zeros(24)],
             'observation_mask': mask, 'observations': np.array([1., 2.]),
             'data_covariance': np.array([2., 8.]), 'final_inflation_factor': 2.,
             'posterior_misfit': [999.], 'posterior_prediction_kind': 'linearized'}
    checkpoint.write_bytes(pickle.dumps(saved))
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    with h5py.File(benchmark / 'globalmodel.h5', 'w') as handle:
        handle['wellpath/Z'] = [[100.]]

    class Adapter:
        def run_pred_multi(self, params, index, shift, well, timeout):
            assert len(params) == 1 and timeout == 45.
            return np.array([[1., 4., 100., 100., 100., 100., 100., 100.]])

    monkeypatch.setattr(check_posterior, 'make_remote2d', lambda *args: Adapter())
    monkeypatch.setattr(sequential, 'dims', (15, 5, 20))
    output = tmp_path / 'check'
    assert check_posterior.main(['--checkpoint', str(checkpoint), '--benchmark', str(benchmark),
                                 '--output-dir', str(output)]) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report['prediction_kind'] == 'forward'
    assert report['members'][0]['forward_misfit'] == pytest.approx(.5)
    assert report['members'][0]['stored_misfit'] == 999.


def test_sample_prior_0d_mixture_statistics(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (15, 5, 20))
    np.random.seed(7)
    background, reservoir, mode_std, fraction = 0.18, 1.46, 0.10, 0.15
    samples, covariance, mean = sequential.sample_prior_0D(4000, background, reservoir,
                                                           mode_std, fraction)
    rh = samples[0]
    reservoirs = rh > 0.5 * (background + reservoir)
    assert abs(reservoirs.mean() - fraction) < 0.03
    assert abs(rh[reservoirs].mean() - reservoir) < 0.02
    assert abs(rh[~reservoirs].mean() - background) < 0.02
    mixture_mean = fraction * reservoir + (1 - fraction) * background
    np.testing.assert_allclose(mean, [mixture_mean, 0.])
    np.testing.assert_allclose(np.diag(covariance)[0], np.var(rh), rtol=0.05)


def test_blocky_prior_switches_and_center_conditioning(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (15, 5, 20))
    np.random.seed(11)
    nz = 20
    background, reservoir, mode_std, fraction = 0.18, 1.46, 0.05, 0.5
    zero_d = np.array([0.2] * 3 + [1.4] * 2)
    ensemble, covariance, mean = sequential.sample_blocky_prior_1D(
        5, zero_d, background, reservoir, mode_std, fraction)
    assert ensemble.shape == (2 * nz, 5)
    midpoint = 0.5 * (background + reservoir)
    for member in range(5):
        rh = ensemble[:nz, member]
        switches = np.flatnonzero((rh[1:] > midpoint) != (rh[:-1] > midpoint))
        assert 1 <= len(switches) <= 4
        assert np.all(np.diff(switches) > 0)
        # the segment containing the center cell follows the 0D posterior side
        center_mode = rh[nz // 2] > midpoint
        assert center_mode == (zero_d[member] > midpoint)
        assert ((rh > midpoint) == center_mode).mean() > 0.  # deterministic sanity
    assert covariance.shape == (2 * nz, 2 * nz)
    assert np.allclose(covariance, np.diag(np.diag(covariance)))


def test_shift_posterior_1d_tracks_and_extends_edges():
    fields = np.zeros((1, 2, 4))
    fields[0, 0, 2] = 7.
    fields[0, 1, 1] = 9.
    result = shift_posterior_1d(fields.reshape(1, -1), 4, (1, -1)).reshape(fields.shape)
    assert result[0, 0, 3] == 7.
    assert result[0, 0, 1] == 0.
    assert result[0, 1, 2] == 9.
    assert fields[0, 0, 2] == 7.


def test_chained_prior_preserves_trend_and_switches(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (15, 5, 20))
    np.random.seed(3)
    nz = 20
    background, reservoir, mode_std, fraction, switch_prob = 0.18, 1.46, 0.10, 0.15, 0.05
    trend = np.zeros((2 * nz, 2))
    trend[:nz, 0] = background
    trend[:nz, 1] = reservoir
    trend[nz:] = 0.1
    ensemble, covariance, mean = sequential.sample_chained_prior_1D(
        50, trend, background, reservoir, mode_std, fraction, switch_prob)
    rh = ensemble[:nz]
    midpoint = 0.5 * (background + reservoir)
    kept = rh[:, :50]  # members 0..49 cycle trends 0,1
    member0 = rh[:, 0::2][:, :25]
    flips0 = (member0 > midpoint).mean()
    flips1 = (rh[:, 1::2][:, :25] < midpoint).mean()
    assert flips0 < 0.2 and flips1 < 0.2
    background_members = rh[:, 0::2].ravel()
    assert np.abs(background_members[background_members < midpoint].mean() - background) < 0.05
    latent = ensemble[nz:]
    assert abs(latent.mean() - 0.1) < 0.05
    assert covariance.shape == (2 * nz, 2 * nz)


def test_soften_column_edges_ramps_blocky_steps():
    nz = 6
    column = np.concatenate([np.full(3, 0.18), np.full(3, 1.46), np.full(nz, 0.25)])
    softened = sequential.soften_column_edges(column, nz)
    jumps = np.abs(np.diff(softened[:nz]))
    assert jumps.max() < (1.46 - 0.18) / 2 + 1e-12
    assert np.allclose(softened[nz:], column[nz:])
    np.testing.assert_allclose(softened[:nz], [0.18, 0.18, 0.5, 1.14, 1.46, 1.46])


def test_project_to_modes_anchors_deviations():
    background, reservoir, radius = 0.18, 1.46, 0.2
    values = np.array([0.18, 1.46, -0.9, 4.0, 0.83, 0.81])
    anchored = project_to_modes(values, background, reservoir, radius)
    np.testing.assert_allclose(anchored[:2], [background, reservoir])
    assert anchored[2] == pytest.approx(background - radius)
    assert anchored[3] == pytest.approx(reservoir + radius)
    assert anchored[4] == pytest.approx(reservoir - radius)
    assert anchored[5] == pytest.approx(background + radius)
    matrix = np.array([[0.0, 2.0], [1.0, 3.0]])
    out = project_to_modes(matrix, background, reservoir, radius)
    assert out.shape == matrix.shape
    assert np.all(np.minimum(np.abs(out - background), np.abs(out - reservoir)) <= radius + 1e-12)
    with pytest.raises(ValueError):
        project_to_modes(values, 1.46, 0.18, radius)
    with pytest.raises(ValueError):
        project_to_modes(values, background, reservoir, 0.)


def test_cli_rejects_invalid_and_full_benchmark_grids():
    with pytest.raises(SystemExit):
        sequential.parse_args(['--output-dir', 'unused', '--nx', '376', '--nz', '801'])
    with pytest.raises(SystemExit):
        sequential.parse_args(['--output-dir', 'unused', '--frequency', '0'])
    with pytest.raises(SystemExit):
        sequential.parse_args(['--output-dir', 'unused', '--typo'])


@pytest.mark.parametrize('force_2d', [False, True])
def test_three_point_controller_carries_window_and_checkpoints(tmp_path, monkeypatch, force_2d):
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    with h5py.File(benchmark / 'globalmodel.h5', 'w') as handle:
        handle['wellpath/Z'] = np.array([[100.], [101.524], [103.048]])
    tool = ('6kHz', '83ft')
    depths = np.array([100., 101.524, 103.048]) * sequential.meter_to_feet
    pd.DataFrame({tool: [np.ones(8)] * 3}, index=depths).to_pickle(tmp_path / 'data.pkl')
    pd.DataFrame({tool: [['ABS', [1.] * 8]] * 3}, index=depths).to_pickle(tmp_path / 'var.pkl')

    def setup(benchmark, tools):
        monkeypatch.setattr(sequential, 'TVD', depths, raising=False)
        monkeypatch.setattr(sequential, 'WX', np.arange(3) * 5., raising=False)
        monkeypatch.setattr(sequential, 'WY', np.zeros(3), raising=False)

    class Backend:
        def trusts_ad(self, fidelity):
            return True

    class Adapter:
        def backend(self):
            return Backend()

        def runtime_identity(self):
            return {'test': True}

    monkeypatch.setattr(sequential, 'setup_simulators', setup)
    monkeypatch.setattr(sequential, '_set_logging_point_state', lambda index: None)
    monkeypatch.setattr(sequential, 'make_remote2d', lambda *args: Adapter())
    monkeypatch.setattr(sequential, 'simulate_pred_only', lambda *args: np.ones(8))
    monkeypatch.setattr(sequential, 'simulate_and_grad_multi',
                        lambda param_list, index: [(np.ones(8), np.zeros((8, len(param))))
                                                   for param in param_list])
    cwd = Path.cwd()
    output = tmp_path / 'run'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--nx', '3', '--nz', '4', '--ensemble-size', '4', '--opt-members', '2',
                     '--max-iterations', '0'] + (['--force-2d'] if force_2d else []))
    assert Path.cwd() == cwd
    expected_params = 2 * (3 * 4 if force_2d else 4)
    for index in range(3):
        result = pd.read_pickle(output / f'inversion_results_assim_{index}.pkl')
        assert np.shape(result['posterior_params']) == (2, expected_params)
        assert result['previous_index'] == (index - 1 if index else None)
        assert result['window_movement'] == ((1, 1) if index else (0, 0))
        assert result['optimized_2d'] == force_2d
        assert result['selected_fidelity'] == ('2D' if force_2d else '1D')
        assert result['two_d_calls'] == (2 if force_2d else 0)
        for member in range(2):
            loss = sequential.custom_loss(result['posterior_predictions'][member],
                                          result['perturbed_observations'][:, member],
                                          result['posterior_params'][member], result['prior_mean'],
                                          result['prior_covariance'], result['data_covariance'])
            assert loss == pytest.approx(result['posterior_losses'][member])


@pytest.fixture
def controller_environment(tmp_path, monkeypatch):
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    with h5py.File(benchmark / 'globalmodel.h5', 'w') as handle:
        handle['wellpath/Z'] = np.array([[100.], [101.524], [103.048]])
    tool = ('6kHz', '83ft')
    depths = np.array([100., 101.524, 103.048]) * sequential.meter_to_feet
    pd.DataFrame({tool: [np.ones(8)] * 3}, index=depths).to_pickle(tmp_path / 'data.pkl')
    pd.DataFrame({tool: [['ABS', [1.] * 8]] * 3}, index=depths).to_pickle(tmp_path / 'var.pkl')

    def setup(benchmark, tools):
        monkeypatch.setattr(sequential, 'TVD', depths, raising=False)
        monkeypatch.setattr(sequential, 'WX', np.arange(3) * 5., raising=False)
        monkeypatch.setattr(sequential, 'WY', np.zeros(3), raising=False)

    class Backend:
        def trusts_ad(self, fidelity):
            return True

    class Adapter:
        def backend(self):
            return Backend()

        def runtime_identity(self):
            return {'test': True}

    monkeypatch.setattr(sequential, 'setup_simulators', setup)
    monkeypatch.setattr(sequential, '_set_logging_point_state', lambda index: None)
    monkeypatch.setattr(sequential, 'make_remote2d', lambda *args: Adapter())
    return tmp_path, benchmark


def test_adaptive_gate_skips_2d_and_runs_final_1d_update(tmp_path, monkeypatch, controller_environment):
    base, benchmark = controller_environment
    calls = {'2d': 0}

    def fake_pred(param, index, simfidelity='2D'):
        if simfidelity == '2D':
            calls['2d'] += 1
        return np.ones(8) if index < 2 else np.ones(8) * 5

    def fake_grad_multi(param_list, index):
        calls['2d'] += len(param_list)
        return [(np.ones(8), np.zeros((8, len(param)))) for param in param_list]

    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    monkeypatch.setattr(sequential, 'simulate_and_grad_multi', fake_grad_multi)
    output = tmp_path / 'run'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--nx', '3', '--nz', '4', '--ensemble-size', '4', '--opt-members', '2',
                     '--max-iterations', '1'])
    for index in range(3):
        result = pd.read_pickle(output / f'inversion_results_assim_{index}.pkl')
        if index < 2:
            assert result['selected_fidelity'] == '1D'
            assert result['two_d_calls'] == 0
            assert result['optimized_2d'] is False
            assert result['mda_weight'] == pytest.approx(1.)
            assert result['misfit']['before_final'] <= result['misfit_threshold']
            assert np.isfinite(result['misfit']['after_final'])
            assert result['final_iteration_worsened'] in (True, False)
            assert np.shape(result['posterior_params']) == (2, 8)
        else:
            assert result['selected_fidelity'] == '2D'
            assert result['optimized_2d'] is True
            assert result['two_d_calls'] >= 2
            assert result['misfit']['before_final'] > result['misfit_threshold']
            assert np.shape(result['posterior_params']) == (2, 24)
    assert calls['2d'] == pd.read_pickle(output / 'inversion_results_assim_2.pkl')['two_d_calls']


def test_worsened_final_iteration_is_reported_not_reverted(tmp_path, monkeypatch, controller_environment):
    base, benchmark = controller_environment
    ne = 4
    state = {'calls': 0}

    def fake_pred(param, index, simfidelity='2D'):
        state['calls'] += 1
        return np.ones(8) * (1 + 0.001 * (state['calls'] // ne))

    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    output = tmp_path / 'run'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--nx', '3', '--nz', '4', '--ensemble-size', str(ne), '--opt-members', '2',
                     '--max-iterations', '1'])
    result = pd.read_pickle(output / 'inversion_results_assim_0.pkl')
    assert result['selected_fidelity'] == '1D'
    assert result['two_d_calls'] == 0
    assert result['final_iteration_worsened'] is True
    assert result['misfit']['after_final'] > result['misfit']['before_final']
    assert result['mda_weight'] == pytest.approx(1.)


def test_cold_start_2d_prior_after_1d_only_point(tmp_path, monkeypatch, controller_environment):
    base, benchmark = controller_environment

    def fake_pred(param, index, simfidelity='2D'):
        return np.ones(8) if index == 0 else np.ones(8) * 5

    def fake_grad_multi(param_list, index):
        return [(np.ones(8), np.zeros((8, len(param)))) for param in param_list]

    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    monkeypatch.setattr(sequential, 'simulate_and_grad_multi', fake_grad_multi)
    output = tmp_path / 'run'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--nx', '3', '--nz', '4', '--ensemble-size', '4', '--opt-members', '2',
                     '--max-iterations', '0'])
    first = pd.read_pickle(output / 'inversion_results_assim_0.pkl')
    second = pd.read_pickle(output / 'inversion_results_assim_1.pkl')
    assert first['selected_fidelity'] == '1D'
    assert second['selected_fidelity'] == '2D'
    assert second['prior_params'] is not None
    assert second['prior_params'].shape == (24, 2)
    assert second['previous_index'] == 0
