from pathlib import Path
import pickle
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
    log_rv = log_rh + np.log(sequential.max_ratio / sequential.min_ratio) * norm.cdf(parameters[12:])
    jac = optimizer_jacobian((weights[0] * np.exp(log_rh))[None],
                             (weights[1] * np.exp(log_rv))[None], latent,
                             sequential.min_ratio, sequential.max_ratio,
                             log_rv_cap=sequential.log_rv_cap, log_rh=log_rh)
    direction = np.random.default_rng(17).normal(size=24)
    step = 1e-5
    fd = (response(parameters + step * direction) - response(parameters - step * direction)) / (2 * step)
    np.testing.assert_allclose(jac @ direction, [fd], rtol=1e-7)


def test_rv_parameterization_uses_tighter_ratio_and_hard_cap(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (1, 1, 2))
    params = np.r_[np.full(2, 4.), np.full(2, 4.)]
    fields, log_rh, latent = sequential._build_param_dict(params)
    assert sequential.max_ratio == 3.
    np.testing.assert_allclose(np.exp(fields['rv']), 80.)
    jac = optimizer_jacobian(np.ones((1, 2)), np.ones((1, 2)), latent,
                             sequential.min_ratio, sequential.max_ratio,
                             log_rv_cap=sequential.log_rv_cap, log_rh=log_rh)
    np.testing.assert_array_equal(jac[:, 2:], 0.)
    np.testing.assert_array_equal(jac[:, :2], 1.)


def test_gaussian_prior_can_be_drawn_without_bit_center_conditioning():
    arguments = (4, None, np.log(1.5), np.log(30.), .1, .15, 3, 5, 5., 5.,
                 80., 75., 35., .12)
    np.random.seed(53)
    unconditioned = sequential.sample_gaussian_prior_2d(*arguments)
    target = np.full(4, .7)
    np.random.seed(53)
    conditioned = sequential.sample_gaussian_prior_2d(
        4, target, *arguments[2:])
    n = 3 * 5
    center = (3 // 2) * 5 + 5 // 2
    assert not np.allclose(unconditioned[center], target)
    np.testing.assert_allclose(conditioned[center], target)
    np.testing.assert_array_equal(unconditioned[n:], conditioned[n:])


def test_gaussian_prior_uses_separate_rh_and_latent_lateral_scales():
    base = (4, None, np.log(1.5), np.log(30.), .1, .15, 3, 5, 5., 5.,
            10., 75., 35., .12)
    np.random.seed(71)
    short_latent = sequential.sample_gaussian_prior_2d(*base, latent_corr_x_ft=20.)
    np.random.seed(71)
    long_latent = sequential.sample_gaussian_prior_2d(*base, latent_corr_x_ft=80.)
    cells = 3 * 5
    np.testing.assert_array_equal(short_latent[:cells], long_latent[:cells])
    assert not np.allclose(short_latent[cells:], long_latent[cells:])


def test_new_lateral_and_multistart_defaults_are_point31_settings(tmp_path):
    args, _ = sequential.parse_args(['--output-dir', str(tmp_path / 'run')])
    assert args.prior_correlation_x_ft == 10.
    assert args.prior_latent_correlation_x_ft == 80.
    assert args.lm_laplace_starts == 5


def test_two_step_esmda_updates_only_valid_ranked_members_twice():
    state = np.array([[.2, .4, .6, .8, 1.],
                      [1., 1.2, 1.4, 1.6, 1.8],
                      [.1, .2, .3, .4, .5],
                      [.6, .7, .8, .9, 1.]])
    selected = np.array([4, 1, 3])
    batch_sizes, update_shapes = [], []

    def forward_batch(models):
        batch_sizes.append(len(models))
        return [None if np.isclose(model[0], .4)
                else np.array([model[0], model[2]]) for model in models]

    def update_step(enX, enY, enE, inflated, threshold):
        update_shapes.append(enX.shape)
        assert threshold == .7
        np.testing.assert_allclose(inflated, [4., 8.])
        return np.full_like(enX, .1)

    result = sequential._run_two_step_esmda(
        state, selected, np.array([1., 2.]), np.array([1., 2.]), .5,
        forward_batch, update_step, np.random.default_rng(13), .7)

    assert result['member_indices'].tolist() == [4, 3]
    assert result['alpha'] == [4., 4.]
    assert result['failures'][0]['member'] == 1
    assert batch_sizes == [3, 2, 2]
    assert update_shapes == [(4, 2), (4, 2)]
    np.testing.assert_allclose(result['state'][:, [4, 3]], state[:, [4, 3]] + .2)
    np.testing.assert_array_equal(result['state'][:, [0, 1, 2]], state[:, [0, 1, 2]])


def test_paired_mean_correction_shifts_remaining_members_and_preserves_their_covariance():
    ensemble_1d = np.array([[.2, .4, .6, .8],
                            [1.0, 1.2, 1.4, 1.6],
                            [.1, .2, .3, .4],
                            [.5, .6, .7, .8]])
    selected = np.array([0, 2])
    selected_2d = ensemble_1d[:, selected] + np.array([[.3], [-.2], [.1], [.4]])
    remaining = np.array([1, 3])
    covariance_before = np.cov(ensemble_1d[:, remaining])

    corrected, delta, damping = sequential._paired_mean_correction(
        ensemble_1d, selected, ensemble_1d[:, selected], selected_2d, rh_cells=2)

    np.testing.assert_allclose(delta, [.3, -.2, .1, .4])
    np.testing.assert_array_equal(damping, np.ones(4))
    np.testing.assert_allclose(corrected[:, selected], selected_2d)
    np.testing.assert_allclose(corrected[:, remaining],
                               ensemble_1d[:, remaining] + delta[:, None])
    np.testing.assert_allclose(np.cov(corrected[:, remaining]), covariance_before)


def test_paired_mean_correction_damps_only_to_respect_rh_bounds():
    ensemble_1d = np.array([[.2, 3.9, .6, 3.8],
                            [1.0, 1.2, 1.4, 1.6],
                            [.1, .2, .3, .4],
                            [.5, .6, .7, .8]])
    selected = np.array([0, 2])
    selected_2d = ensemble_1d[:, selected] + np.array([[1.], [0.], [0.], [0.]])

    corrected, delta, damping = sequential._paired_mean_correction(
        ensemble_1d, selected, ensemble_1d[:, selected], selected_2d, rh_cells=2)

    assert damping[0] == pytest.approx(.1)
    np.testing.assert_array_equal(damping[1:], np.ones(3))
    assert np.max(corrected[:2]) <= sequential.LOG_RH_LIMITS[1]
    assert np.min(corrected[:2]) >= sequential.LOG_RH_LIMITS[0]
    np.testing.assert_allclose(corrected[2:, [1, 3]],
                               ensemble_1d[2:, [1, 3]]
                               + damping[2:, None] * delta[2:, None])


def test_auto_adaloc_sigmoid_midpoint_is_requested_threshold():
    from pipt.misc_tools.cov_regularization import localization

    config = sequential._auto_adaloc_config(.7, 12)
    assert config == {'field': [12, 1, 1], 'autoadaloc': .3, 'type': 'sigm'}
    localizer = localization(config, [], [], ['rh', 'rv'], 4)
    correlations = np.array([[.69], [.7], [.71]])
    taper = localizer.tapering_function(correlations, np.zeros_like(correlations))
    assert taper[0, 0] < .5 < taper[2, 0]
    assert taper[1, 0] == pytest.approx(.5)


def test_two_step_esmda_slurm_raises_batch_concurrency_to_member_count(tmp_path):
    args, config = sequential.parse_args([
        '--output-dir', str(tmp_path / 'run'), '--final-stage', 'two-step-esmda',
        '--sim-backend', 'slurm', '--concurrency', '4'])
    assert args.final_2d_members == 10
    assert args.final_2d_localization_threshold == .8
    assert config.concurrency == 10


def test_adaptive_localization_flag_keeps_final_2d_alias(tmp_path):
    args, _ = sequential.parse_args([
        '--output-dir', str(tmp_path / 'run'), '--adaptive-localization-threshold', '.65'])
    assert args.final_2d_localization_threshold == .65
    args, _ = sequential.parse_args([
        '--output-dir', str(tmp_path / 'run2'), '--final-2d-localization-threshold', '.55'])
    assert args.final_2d_localization_threshold == .55


def test_two_step_esmda_controller_runs_gated_forward_only_batches(
        tmp_path, monkeypatch, controller_environment):
    _, benchmark = controller_environment
    calls = {'batches': [], 'jacobian': 0}

    def fake_pred(param, index, simfidelity='2D'):
        return np.ones(8) if simfidelity == '0D' else np.full(8, 5.)

    class Backend:
        def trusts_ad(self, fidelity):
            return True

    class Adapter:
        def backend(self):
            return Backend()

        def runtime_identity(self):
            return {'test': True}

        def run_pred_multi(self, param_dicts, index, shift, well, timeout=None):
            calls['batches'].append(len(param_dicts))
            return np.asarray([
                np.full(8, float(np.mean(parameters['rh'])))
                for parameters in param_dicts])

    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    monkeypatch.setattr(sequential, 'make_remote2d', lambda *args: Adapter())
    output = tmp_path / 'two-step'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--stop', '1', '--nx', '3', '--nz', '4', '--ensemble-size', '4',
                     '--opt-members', '3',
                     '--final-stage', 'two-step-esmda', '--final-2d-members', '3',
                     '--prior-style', 'gaussian', '--prior-projection', 'off'])

    result = pd.read_pickle(output / 'inversion_results_assim_0.pkl')
    assert result['final_stage'] == 'two-step-esmda'
    assert result['misfit']['before_final'] > result['misfit_threshold']
    assert len(result['final_2d_ranked_candidates']) == 3
    assert result['final_2d_alpha'][0] == result['final_2d_alpha'][1]
    assert result['final_2d_localization'] == {
        'type': 'sigm', 'correlation_threshold': .8, 'autoadaloc': .2}
    assert result['adaptive_localization'] == result['final_2d_localization']
    assert result['mean_correction_method'] == 'paired_selected_2d_minus_1d'
    assert len(result['mean_correction_delta']) == 24
    assert np.asarray(result['mean_correction_damping']).mean() > 0.
    assert calls['batches'] == [3, 3, 3]
    assert result['two_d_calls'] == 9
    assert result['simulator_calls']['jacobian_2D'] == 0
    assert np.shape(result['posterior_params']) == (3, 24)
    assert np.shape(result['posterior_ensemble_params']) == (4, 24)


def test_two_step_esmda_respects_closed_1d_to_2d_gate(
        tmp_path, monkeypatch, controller_environment):
    _, benchmark = controller_environment
    prior_covariance_shapes = []
    original_loss = sequential.custom_loss

    def track_loss(predictions, observations, theta, theta_mean, covariance, data_covariance):
        prior_covariance_shapes.append(np.asarray(covariance).shape)
        return original_loss(predictions, observations, theta, theta_mean,
                             covariance, data_covariance)

    monkeypatch.setattr(sequential, 'simulate_pred_only',
                        lambda *args: np.ones(8))
    monkeypatch.setattr(sequential, 'custom_loss', track_loss)
    output = tmp_path / 'gate-closed'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--stop', '1', '--nx', '3', '--nz', '4', '--ensemble-size', '4',
                     '--opt-members', '3', '--final-stage', 'two-step-esmda',
                     '--prior-style', 'gaussian', '--prior-projection', 'off'])

    result = pd.read_pickle(output / 'inversion_results_assim_0.pkl')
    assert result['selected_fidelity'] == '1D'
    assert result['two_d_calls'] == 0
    assert result['final_stage_skipped'] == '1D-to-2D threshold gate remained closed'
    assert prior_covariance_shapes
    assert all(len(shape) == 1 for shape in prior_covariance_shapes)


def test_resume_state_restores_previous_posterior_and_window_offset(tmp_path):
    dims, spacing = (3, 5, 4), (5., 5., 5.)
    tools = [('6kHz', '83ft')]
    wx, wy, tvd = np.array([0., 5., 10.]), np.zeros(3), np.array([100., 100., 100.])
    window = LocalWindow(dims, spacing)
    window.update((wx[0], wy[0], tvd[0]))
    shift, _ = window.update((wx[1], wy[1], tvd[1]))
    posterior = np.arange(2 * np.prod(dims[::2]) * 2., dtype=float).reshape(2, -1)
    saved = {'logging_index': 1, 'dims': dims, 'spacing_ft': spacing,
             'tools': tools, 'prior_style': 'gaussian',
             'rv_ratio_bounds': (1., 3.), 'rv_hard_cap_ohmm': 80.,
             'laplace_correlation_ft': [(10., 75.), (80., 35.)],
             'posterior_params': list(posterior), 'shift_ft': shift}
    checkpoint = tmp_path / 'inversion_results_assim_1.pkl'
    with checkpoint.open('wb') as stream:
        pickle.dump(saved, stream)

    restored, previous_index, digest = sequential.load_resume_state(
        checkpoint, dims, spacing, 2, 2, tools, 'gaussian', (1., 3.), 80.,
        ((10., 75.), (80., 35.)), LocalWindow(dims, spacing), wx, wy, tvd)
    np.testing.assert_array_equal(restored, posterior.T)
    assert previous_index == 1
    assert len(digest) == 64


def test_controller_warm_resumes_from_previous_checkpoint(tmp_path, monkeypatch,
                                                          controller_environment):
    base, benchmark = controller_environment
    dims, spacing = (3, 5, 4), (5., 5., 5.)
    wx, wy = np.arange(3) * 5., np.zeros(3)
    tvd = np.array([100., 101.524, 103.048]) * sequential.meter_to_feet
    window = LocalWindow(dims, spacing)
    shift, movement = window.update((wx[0], wy[0], tvd[0]))
    assert movement == (0, 0)
    ensemble = np.zeros((2 * dims[0] * dims[2], 6))
    checkpoint = base / 'inversion_results_assim_0.pkl'
    with checkpoint.open('wb') as stream:
        pickle.dump({'logging_index': 0, 'dims': dims, 'spacing_ft': spacing,
                     'tools': [('6kHz', '83ft')], 'prior_style': 'gaussian',
                     'rv_ratio_bounds': (1., 3.), 'rv_hard_cap_ohmm': 80.,
                     'laplace_correlation_ft': [(10., 75.), (80., 35.)],
                     'posterior_params': list(ensemble.T), 'shift_ft': shift}, stream)
    monkeypatch.setattr(sequential, 'simulate_pred_only', lambda *args: np.ones(8))
    output = base / 'resumed'
    sequential.main(['--benchmark', str(benchmark), '--data', str(base / 'data.pkl'),
                     '--variance', str(base / 'var.pkl'), '--output-dir', str(output),
                     '--start', '1', '--stop', '3', '--nx', '3', '--nz', '4',
                     '--ensemble-size', '6', '--prior-style', 'gaussian',
                     '--prior-projection', 'off', '--resume-checkpoint', str(checkpoint)])
    first = pd.read_pickle(output / 'inversion_results_assim_1.pkl')
    assert first['previous_index'] == 0
    assert first['window_movement'] == (1, 1)
    assert first['effective_random_seed'] == 2027
    manifest = __import__('json').loads((output / 'run.json').read_text())
    assert manifest['resume_checkpoint'] == str(checkpoint.resolve())
    assert len(manifest['resume_checkpoint_sha256']) == 64


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
    assert meta['orientation'] == 'gradient'


def test_material_orientation_flows_from_cli_and_environment_into_metadata(monkeypatch):
    import argparse

    parser = argparse.ArgumentParser()
    SimBackendConfig().add_cli(parser)
    config = SimBackendConfig().from_args(parser.parse_args(['--orientation', 'bedding']))
    assert config.orientation == 'bedding' and config.to_env()['DA_SIM_ORIENTATION'] == 'bedding'
    with pytest.raises(SystemExit):
        parser.parse_args(['--orientation', 'tilted'])
    monkeypatch.setenv('DA_SIM_ORIENTATION', 'bedding')
    assert SimBackendConfig.from_env().orientation == 'bedding'
    adapter = Remote2D(config, (3, 5, 4), 5., 5., 5., [('6kHz', '83ft')])
    well = {key: np.array([[0.]]) for key in ('X', 'Y', 'Z', 'Distance', 'Inclination', 'Azimuth')}
    assert adapter.build_metadata(0, {'x': 0., 'y': 0., 'z': 0.}, well)['orientation'] == 'bedding'


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


def test_custom_loss_vector_diagonal_covariance_matches_matrix():
    predictions = np.array([1., 3.])
    observations = np.array([2., 2.])
    theta = np.array([1., -1., .5])
    mean = np.array([0., 0., 0.])
    diagonal = np.array([2., 4., 8.])
    cd = np.array([1., 2.])
    vector_loss = sequential.custom_loss(predictions, observations, theta, mean, diagonal, cd)
    matrix_loss = sequential.custom_loss(predictions, observations, theta, mean,
                                         np.diag(diagonal), cd)
    assert vector_loss == pytest.approx(matrix_loss)


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
        assert np.shape(result['posterior_params']) == (2, 24)
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


def test_one_d_budget_reserves_final_likelihood_weight():
    args, _ = sequential.parse_args(['--output-dir', 'unused', '--one-d-budget', '.25'])
    assert args.one_d_budget == pytest.approx(.25)
    for bad in ('0', '-.1', '.8', 'nan'):
        with pytest.raises(SystemExit):
            sequential.parse_args(['--output-dir', 'unused', '--one-d-budget', bad])


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
        monkeypatch.setattr(sequential, 'wellpath', {}, raising=False)

    class Backend:
        def trusts_ad(self, fidelity):
            return True

    class Adapter:
        def backend(self):
            return Backend()

        def runtime_identity(self):
            return {'test': True}

        def run_pred_multi(self, param_dicts, index, shift, well, timeout=None):
            return np.full((len(param_dicts), len(sequential.UDAR_COMPONENTS) * len(sequential.tools)),
                           np.nan)

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
    expected_params = 2 * 3 * 4
    for index in range(3):
        result = pd.read_pickle(output / f'inversion_results_assim_{index}.pkl')
        assert np.shape(result['posterior_params']) == (2, expected_params)
        assert result['unified_domain'] is True
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
        monkeypatch.setattr(sequential, 'wellpath', {}, raising=False)

    class Backend:
        def trusts_ad(self, fidelity):
            return True

    class Adapter:
        def backend(self):
            return Backend()

        def runtime_identity(self):
            return {'test': True}

        def run_pred_multi(self, param_dicts, index, shift, well, timeout=None):
            return np.full((len(param_dicts), len(sequential.UDAR_COMPONENTS) * len(sequential.tools)),
                           np.nan)

    monkeypatch.setattr(sequential, 'setup_simulators', setup)
    monkeypatch.setattr(sequential, '_set_logging_point_state', lambda index: None)
    monkeypatch.setattr(sequential, 'make_remote2d', lambda *args: Adapter())
    return tmp_path, benchmark


def test_lm_laplace_controller_checkpoints_after_five_batched_2d_start_failures(
        tmp_path, monkeypatch, controller_environment):
    _, benchmark = controller_environment

    def fake_pred(param, index, simfidelity='2D'):
        if simfidelity == '2D':
            raise AssertionError('LM 2D starts should use the batched remote adapter')
        return np.ones(8) * 5.

    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    output = tmp_path / 'run'
    sequential.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
                     '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
                     '--nx', '3', '--nz', '4', '--ensemble-size', '6', '--opt-members', '2',
                     '--max-iterations', '0', '--lm-laplace-iterations', '1',
                     '--prior-style', 'gaussian', '--prior-projection', 'off',
                     '--final-stage', 'lm-laplace'])
    for index in range(3):
        result = pd.read_pickle(output / f'inversion_results_assim_{index}.pkl')
        assert result['gate_selected_fidelity'] == '2D'
        assert result['selected_fidelity'] == '1D'
        assert result['two_d_calls'] == 5
        assert len(result['solver_failures']) == 5
        assert not result['invariant_start_attempted']
        assert result['solver_failures'] == result['two_d_initial_failures']
        assert result['solver_failures'][0]['start_kind'] == 'ranked_member'
        assert result['solver_failures'][0]['reason'] == 'batch_forward_failure'
        assert result['posterior_forward_verified']
        assert result['previous_index'] == (index - 1 if index else None)


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
            assert np.shape(result['posterior_params']) == (2, 24)
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
