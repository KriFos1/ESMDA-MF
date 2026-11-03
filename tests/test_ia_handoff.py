import importlib.util
import json
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import chain_laplace
from field_trajectory import read_trajectory
import pilot_rml_index45 as rml
import run_ia_chain as ia
from run_ia_chain import lwd_means


def ia_prepare_data():
    spec = importlib.util.spec_from_file_location('ia_prepare_data', ia.ROOT.parent / 'IA/prepare_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_ia_simulators(monkeypatch):
    """1D predictions of ln Rh(0) + dip; every update lowers ln Rh(0) by 10 and raises the dip latent by .4."""
    class FakeEnIF:
        def __init__(self, *args):
            pass
        def step(self, state, predicted, perturbed):
            step = np.zeros_like(state)
            step[0] = -10.
            step[-1] = .4
            return step

    monkeypatch.setattr(ia, 'EnIFProbe', FakeEnIF)
    for name in ('dims', 'dX', 'dY', 'dZ', 'tools', 'anisotropy', 'log_rv_cap', 'min_ratio', 'max_ratio',
                 'current_shift'):
        monkeypatch.setattr(ia.sequential, name, getattr(ia.sequential, name, None), raising=False)
    monkeypatch.setattr(ia.sequential, 'setup_simulators', lambda *args, **kwargs: None)
    monkeypatch.setattr(ia.sequential, 'simulate_pred_only',
                        lambda values, index, fidelity, apparent_dip: np.full(72, values[0] + apparent_dip))


def test_lwd_prior_mean_uses_both_values_without_resistivity_bounds(tmp_path):
    path = tmp_path / 'lwd.txt'
    path.write_text('% logpt Res1 Res2\n1 0.1 0.4\n2 100 900\n')
    np.testing.assert_allclose(np.exp(lwd_means(path, 2)), [.2, 300.])
    logs = ia.lwd_log_values(path, 2)
    np.testing.assert_allclose(logs.mean(), np.mean(np.log([.1, .4, 100., 900.])))
    np.testing.assert_allclose(logs.std(), np.std(np.log([.1, .4, 100., 900.])))


def test_field_gate_map_and_laplace_keep_dip_and_unclipped_fields(tmp_path):
    chain = tmp_path / 'chain'
    chain.mkdir()
    (chain / 'study.json').write_text(json.dumps({'method': 'ia_field_dip_enif'}))
    rng = np.random.default_rng(3)
    nx, nz, members = 4, 5, 10
    cells = nx * nz
    fields = rng.normal(0., .2, (2 * cells, members))
    fields[:cells // 2] += np.log(.1)
    fields[cells // 2:cells] += np.log(100.)
    dip = rng.normal(0., .2, members)
    mask = np.ones(8, dtype=bool)
    probe = {'open': True, 'reverted': True, 'threshold': 25.,
             'pre_update_mean_prediction_misfit': 10., 'post_mean_prediction_misfit': 30.}
    problem = chain / 'laplace-problem-000.npz'
    well = {key: np.zeros((1, 1)) for key in ('X', 'Y', 'Z', 'Distance', 'Inclination', 'Azimuth')}
    chain_laplace.export_problem(problem, index=0, state=fields, observations=np.ones(8), variance=np.ones(8),
        mask=mask, shift={'x': 0., 'y': 0., 'z': 0.}, movement=(0, 0), dims=(nx, 5, nz),
        spacing=(5., 5., 5.), tools=[('24kHz', '15.61ft')], anisotropy='log',
        anisotropy_moments=(0., .3), ranges=[[50., 50.], [80., 35.]], probe=probe,
        well=well, prior_type='ensemble_precision', dip_latent=dip)
    (chain / 'gate-000.json').write_text(json.dumps({'logging_index': 0, 'previous_index': None,
                                                      'problem': problem.name, 'probe': probe}))
    source = Path(rml.ROOT) / 'references/map-source'
    with pytest.raises(ValueError, match='concurrency of at least 4'):
        rml.prepare_laplace(SimpleNamespace(run_dir=tmp_path / 'insufficient', source=source,
                            binary_dir=None, orientation=None, problem=problem, node_solver=(1, 1)))
    assert not (tmp_path / 'insufficient').exists()
    output = tmp_path / 'map'
    rml.prepare_laplace(SimpleNamespace(run_dir=output, source=source, binary_dir=None, orientation=None,
                                       problem=problem))
    study = json.loads((output / 'study.json').read_text())
    assert study['rh_limits'] is None and study['log_rv_cap'] is None
    assert study['forward_only_candidates'] is True and study['optimizer']['candidates'] == 4
    assert study['count'] == 1
    prior = rml.load_prior(study, output)
    assert len(prior.mean) == 2 * cells + 1
    member = output / 'member-000'
    member.mkdir()
    prediction = np.ones(8)
    np.savez(member / 'fit.npz', model=prior.mean, z=np.zeros(len(prior.mean)), prediction=prediction)
    np.savez(member / 'laplace.npz', prediction=prediction, jacobian=np.zeros((8, len(prior.mean))))
    (member / 'result.json').write_text(json.dumps({'reason': 'iteration_limit'}))
    chain_laplace.sample(SimpleNamespace(chain=chain, map=output, count=members, seed=5))
    with (chain / 'inversion_results_assim_0.pkl').open('rb') as stream:
        checkpoint = pickle.load(stream)
    assert checkpoint['post_mda_params'].shape == (2 * cells, members)
    assert checkpoint['dip_latent'].shape == (members,)
    assert checkpoint['rh_cap_ohmm'] is None and checkpoint['rv_hard_cap_ohmm'] is None
    assert np.median(checkpoint['post_mda_params'][:cells // 2]) < np.log(.25)
    assert np.median(checkpoint['post_mda_params'][cells // 2:cells]) > np.log(40.)


def test_fixed_zero_dip_gate_map_and_laplace_are_field_only_and_unclipped(tmp_path):
    chain = tmp_path / 'chain'
    chain.mkdir()
    (chain / 'study.json').write_text(json.dumps({'method': 'ia_field_dip_enif', 'fixed_zero_dip': True}))
    rng = np.random.default_rng(3)
    nx, nz, members = 4, 5, 10
    cells = nx * nz
    fields = rng.normal(0., .2, (2 * cells, members))
    fields[:cells // 2] += np.log(.1)
    fields[cells // 2:cells] += np.log(100.)
    probe = {'open': True, 'reverted': True, 'threshold': 25.,
             'pre_update_mean_prediction_misfit': 10., 'post_mean_prediction_misfit': 30.}
    problem = chain / 'laplace-problem-000.npz'
    well = {key: np.zeros((1, 1)) for key in ('X', 'Y', 'Z', 'Distance', 'Inclination', 'Azimuth')}
    chain_laplace.export_problem(problem, index=0, state=fields, observations=np.ones(8), variance=np.ones(8),
        mask=np.ones(8, dtype=bool), shift={'x': 0., 'y': 0., 'z': 0.}, movement=(0, 0), dims=(nx, 5, nz),
        spacing=(5., 5., 5.), tools=[('24kHz', '15.61ft')], anisotropy='log',
        anisotropy_moments=(0., .3), ranges=[[50., 50.], [80., 35.]], probe=probe,
        well=well, prior_type='ensemble_precision', fixed_zero_dip=True)
    (chain / 'gate-000.json').write_text(json.dumps({'logging_index': 0, 'previous_index': None,
                                                      'problem': problem.name, 'probe': probe}))
    exported = chain_laplace.load_problem(problem)
    assert exported['fixed_zero_dip'] and 'dip_latent' not in exported
    output = tmp_path / 'map'
    rml.prepare_laplace(SimpleNamespace(run_dir=output, source=Path(rml.ROOT) / 'references/map-source',
                                       binary_dir=None, orientation=None, problem=problem))
    study = json.loads((output / 'study.json').read_text())
    assert study['rh_limits'] is None and study['log_rv_cap'] is None
    assert 'dip_prior' not in study and study['forward_only_candidates'] is True
    prior = rml.load_prior(study, output)
    assert len(prior.mean) == 2 * cells
    member = output / 'member-000'
    member.mkdir()
    prediction = np.ones(8)
    np.savez(member / 'fit.npz', model=prior.mean, z=np.zeros(len(prior.mean)), prediction=prediction)
    np.savez(member / 'laplace.npz', prediction=prediction, jacobian=np.zeros((8, len(prior.mean))))
    (member / 'result.json').write_text(json.dumps({'reason': 'iteration_limit'}))
    chain_laplace.sample(SimpleNamespace(chain=chain, map=output, count=members, seed=5))
    with (chain / 'inversion_results_assim_0.pkl').open('rb') as stream:
        checkpoint = pickle.load(stream)
    assert checkpoint['post_mda_params'].shape == (2 * cells, members)
    assert 'dip_latent' not in checkpoint
    assert checkpoint['rh_cap_ohmm'] is None and checkpoint['rv_hard_cap_ohmm'] is None
    assert np.median(checkpoint['post_mda_params'][:cells // 2]) < np.log(.25)
    assert np.median(checkpoint['post_mda_params'][cells // 2:cells]) > np.log(40.)


def test_fixed_zero_driver_carries_field_only_and_rejects_dip_mode_change(tmp_path, monkeypatch):
    fake_ia_simulators(monkeypatch)
    output = tmp_path / 'chain'
    common = ['--output-dir', str(output), '--nx', '3', '--nz', '4', '--ensemble-size', '4',
              '--mda-steps', '1', '--gate-threshold', '1e300']
    assert ia.main([*common, '--stop', '1', '--fixed-zero-dip']) == 0
    study = json.loads((output / 'study.json').read_text())
    assert study['fixed_zero_dip'] and '--fixed-zero-dip' in ia.driver_args(study, 2)
    with (output / 'inversion_results_assim_0.pkl').open('rb') as stream:
        first = pickle.load(stream)
    assert first['post_mda_params'].shape == (24, 4) and 'dip_latent' not in first
    assert ia.main([*common, '--stop', '2', '--fixed-zero-dip', '--resume']) == 0
    with pytest.raises(ValueError, match='Cannot change dip mode'):
        ia.main([*common, '--stop', '3', '--resume'])


def test_ia_driver_updates_and_carries_dip_without_clipping(tmp_path, monkeypatch):
    fake_ia_simulators(monkeypatch)
    output = tmp_path / 'chain'
    common = ['--output-dir', str(output), '--nx', '3', '--nz', '4', '--ensemble-size', '4',
              '--mda-steps', '1', '--gate-threshold', '1e300']
    assert ia.main([*common, '--stop', '1']) == 0
    with (output / 'inversion_results_assim_0.pkl').open('rb') as stream:
        first = pickle.load(stream)
    np.testing.assert_allclose(first['dip_latent'], first['prior_dip_latent'] + .4)
    assert np.all(first['post_mda_params'][0] < -1.)
    assert first['rv_hard_cap_ohmm'] is None and first['rh_cap_ohmm'] is None
    assert ia.main([*common, '--stop', '2', '--resume']) == 0
    with (output / 'inversion_results_assim_1.pkl').open('rb') as stream:
        second = pickle.load(stream)
    np.testing.assert_allclose(second['dip_latent'], second['prior_dip_latent'] + .4)
    study = json.loads((output / 'study.json').read_text())
    history = json.loads((output / 'history.json').read_text())
    assert study['lwd_prior_mode'] == 'section-global'
    assert study['lwd_prior_sample_count'] == 758
    assert history[0]['lwd_prior_mean_ohmm'] == pytest.approx(history[1]['lwd_prior_mean_ohmm'])
    logs = ia.lwd_log_values(ia.INPUTS / 'LWD_resistivity.txt', 379)
    assert study['lwd_prior_log_mean'] == pytest.approx(float(logs.mean()))
    assert study['lwd_prior_log_std'] == pytest.approx(float(logs.std()))
    with pytest.raises(ValueError, match='Cannot change LWD prior mode'):
        ia.main([*common, '--stop', '3', '--resume', '--lwd-prior', 'pointwise'])
    legacy = tmp_path / 'legacy'
    legacy_args = ['--output-dir', str(legacy), '--nx', '3', '--nz', '4', '--ensemble-size', '4',
                   '--mda-steps', '1', '--gate-threshold', '1e300']
    assert ia.main([*legacy_args, '--stop', '1', '--lwd-prior', 'pointwise']) == 0
    assert ia.main([*legacy_args, '--stop', '2', '--resume']) == 0
    legacy_history = json.loads((legacy / 'history.json').read_text())
    assert legacy_history[0]['lwd_prior_mean_ohmm'] != pytest.approx(legacy_history[1]['lwd_prior_mean_ohmm'])
    # The shipped tables predate recorded noise rules: they are the paper's 0.5% channel-std model.
    assert study['noise'] == ia.LEGACY_NOISE and history[0]['masked_channels'] == 0


def test_default_noise_rule_reproduces_the_shipped_channel_std_tables(tmp_path):
    prepare = ia_prepare_data()
    prepare.prepare(ia.INPUTS, tmp_path / 'data')
    shipped, rebuilt = (pd.read_pickle(path / 'var.pkl') for path in (ia.DATA, tmp_path / 'data'))
    assert rebuilt.attrs['noise_rule'] == 'channel-std' and rebuilt.attrs['noise_fraction'] == .005
    assert list(rebuilt.columns) == list(shipped.columns)
    for tool in shipped.columns:
        assert all(np.allclose(new[1], old[1], rtol=1e-12, atol=0) for new, old in zip(rebuilt[tool], shipped[tool]))


def test_cross_tool_signal_noise_is_shared_by_the_tools_and_validated(tmp_path, monkeypatch):
    prepare = ia_prepare_data()
    data_dir = tmp_path / 'cross-tool'
    assert prepare.prepare(ia.INPUTS, data_dir, noise_rule='cross-tool-signal') == (379, 9, 0)  # nothing masked
    data, variance = (pd.read_pickle(data_dir / name) for name in ('data.pkl', 'var.pkl'))
    values = np.stack([np.stack(data[tool].to_numpy()) for tool in data.columns], axis=1)
    sigma = np.sqrt(np.stack([np.stack([row[1] for row in variance[tool]]) for tool in variance.columns], axis=1))
    # Each component's sigma is 2.5% of its mean |value| over the tools at that point, for every tool.
    np.testing.assert_allclose(sigma, np.broadcast_to(.025 * np.abs(values).mean(axis=1, keepdims=True), sigma.shape),
                               rtol=1e-12)
    trajectory = read_trajectory(ia.INPUTS / 'trajectory.DAT')
    ia.validate_observations(ia.INPUTS, data, variance, trajectory)
    variance.attrs['noise_rule'] = 'per-tool-signal'
    with pytest.raises(ValueError, match='per-tool-signal noise rule'):
        ia.validate_observations(ia.INPUTS, data, variance, trajectory)
    fake_ia_simulators(monkeypatch)
    output = tmp_path / 'chain'
    assert ia.main(['--output-dir', str(output), '--data-dir', str(data_dir), '--nx', '3', '--nz', '4',
                    '--ensemble-size', '4', '--mda-steps', '1', '--gate-threshold', '1e300', '--stop', '1']) == 0
    assert json.loads((output / 'study.json').read_text())['noise'] == {
        'rule': 'cross-tool-signal', 'fraction': .025, 'noise_floor': 1e-3}
    assert json.loads((output / 'history.json').read_text())[0]['masked_channels'] == 0


def test_per_tool_signal_noise_is_validated_masked_only_below_the_floor_and_fixed_on_resume(tmp_path, monkeypatch):
    prepare = ia_prepare_data()
    data_dir = tmp_path / 'per-tool'
    points, tools, below = prepare.prepare(ia.INPUTS, data_dir, noise_rule='per-tool-signal')
    data, variance = (pd.read_pickle(data_dir / name) for name in ('data.pkl', 'var.pkl'))
    values = np.stack([np.stack(data[tool].to_numpy()) for tool in data.columns], axis=1)
    sigma = np.sqrt(np.stack([np.stack([row[1] for row in variance[tool]]) for tool in variance.columns], axis=1))
    np.testing.assert_allclose(sigma, np.maximum(.025 * np.abs(values), 1e-12), rtol=1e-12)
    assert (points, tools) == (379, 9) and 0 < below == np.count_nonzero(sigma < 1e-3)
    assert json.loads((data_dir / 'provenance.json').read_text())['values_below_noise_floor'] == below
    trajectory = read_trajectory(ia.INPUTS / 'trajectory.DAT')
    ia.validate_observations(ia.INPUTS, data, variance, trajectory)
    variance.attrs['noise_rule'] = 'channel-std'
    with pytest.raises(ValueError, match='channel-std noise rule'):
        ia.validate_observations(ia.INPUTS, data, variance, trajectory)

    fake_ia_simulators(monkeypatch)
    output = tmp_path / 'chain'
    common = ['--output-dir', str(output), '--data-dir', str(data_dir), '--nx', '3', '--nz', '4',
              '--ensemble-size', '4', '--mda-steps', '1', '--gate-threshold', '1e300']
    assert ia.main([*common, '--stop', '2']) == 0
    assert json.loads((output / 'study.json').read_text())['noise'] == {
        'rule': 'per-tool-signal', 'fraction': .025, 'noise_floor': 1e-3}
    history = json.loads((output / 'history.json').read_text())
    for index in range(2):
        with (output / f'inversion_results_assim_{index}.pkl').open('rb') as stream:
            result = pickle.load(stream)
        # Only channels below the noise floor are masked; USDP/USDA are kept everywhere else.
        kept = sigma[index].ravel() >= 1e-3
        np.testing.assert_array_equal(result['observation_mask'], kept)
        np.testing.assert_allclose(result['data_covariance'], sigma[index].ravel()[kept] ** 2)
        assert history[index]['masked_channels'] == np.count_nonzero(~kept) > 0
    shutil.rmtree(data_dir)
    prepare.prepare(ia.INPUTS, data_dir)                              # same path, the channel-std rule
    with pytest.raises(ValueError, match='differs in noise'):
        ia.main([*common, '--stop', '3', '--resume'])


def test_ia_launcher_initializes_then_runs_matching_controller(tmp_path, monkeypatch):
    import run_ia_case

    output = tmp_path / 'case'
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            output.mkdir()
            (output / 'study.json').write_text(json.dumps({'method': 'ia_field_dip_enif',
                                                           'ensemble_size': 8,
                                                           'lwd_prior_mode': 'section-global'}))
    monkeypatch.setattr(run_ia_case.subprocess, 'run', run)
    monkeypatch.setattr(run_ia_case, 'implementation_id', lambda: '1234567890')
    assert run_ia_case.main(['--run-dir', str(output), '--stop', '2', '--ensemble-size', '8']) == 0
    assert commands[0][3].endswith('run_ia_chain.py')
    assert commands[1][3].endswith('run_chain_laplace_loop.py')
    assert commands[1][commands[1].index('--count') + 1] == '8'
    assert commands[1][commands[1].index('--orientation') + 1] == 'bedding'
    assert commands[1][commands[1].index('--map-starts') + 1] == '0'
    label = commands[1][commands[1].index('--label') + 1]
    assert label.startswith('ia-case-1234567890-b') and label.endswith('-m0')
    assert '--solver-grid' not in commands[1] and '--solver-tolerance' not in commands[1]
    assert run_ia_case.main(['--run-dir', str(output), '--stop', '2', '--ensemble-size', '8',
                             '--solver-grid', '40', '4', '80', '30', '20', '30', '2', '1', '2',
                             '--solver-tolerance', '1e-5', '--evaluation-timeout', '1500']) == 0
    assert len(commands) == 3                     # the existing chain only starts the controller
    loop = commands[2]
    grid = loop.index('--solver-grid')
    assert loop[grid + 1:grid + 10] == ['40', '4', '80', '30', '20', '30', '2', '1', '2']
    assert loop[loop.index('--solver-tolerance') + 1] == '1e-05'
    assert loop[loop.index('--evaluation-timeout') + 1] == '1500.0'
    # A different solver configuration gets its own MAP study label.
    assert loop[loop.index('--label') + 1].startswith(label + '-s')


def test_prior_ranges_and_warm_start_settings_reach_the_chain_and_the_controller(tmp_path, monkeypatch):
    fake_ia_simulators(monkeypatch)
    output = tmp_path / 'chain'
    common = ['--output-dir', str(output), '--nx', '3', '--nz', '4', '--ensemble-size', '4', '--mda-steps', '1',
              '--gate-threshold', '1e300']
    assert ia.main([*common, '--stop', '1', '--rh-ranges', '100', '50', '--latent-ranges', '160', '35']) == 0
    study = json.loads((output / 'study.json').read_text())
    assert study['ranges'] == [[100., 50.], [160., 35.]]
    arguments = ia.driver_args(study, 2)
    assert arguments[arguments.index('--rh-ranges') + 1:arguments.index('--rh-ranges') + 3] == ['100.0', '50.0']
    assert arguments[arguments.index('--latent-ranges') + 1:arguments.index('--latent-ranges') + 3] == ['160.0', '35.0']
    with pytest.raises(ValueError, match='differs in prior ranges'):
        ia.main([*common, '--stop', '2', '--resume'])
    assert ia.main([*common, '--stop', '2', '--resume', '--rh-ranges', '100', '50', '--latent-ranges', '160', '35']) == 0

    import run_ia_case
    case = tmp_path / 'case'
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            case.mkdir()
            (case / 'study.json').write_text(json.dumps({'method': 'ia_field_dip_enif', 'ensemble_size': 8,
                                                         'lwd_prior_mode': 'section-global'}))
    monkeypatch.setattr(run_ia_case.subprocess, 'run', run)
    monkeypatch.setattr(run_ia_case, 'implementation_id', lambda: '1234567890')
    assert run_ia_case.main(['--run-dir', str(case), '--stop', '5', '--ensemble-size', '8', '--rh-ranges', '100', '50',
                             '--node-sbatch', '8', '4', '--first-map-seconds', '1200',
                             '--optimizer-setting', 'max_seconds=120', '--optimizer-setting', 'initial_trust=2.0']) == 0
    chain_command, loop_command = commands
    assert chain_command[chain_command.index('--rh-ranges') + 1:chain_command.index('--rh-ranges') + 3] == ['100.0', '50.0']
    assert loop_command[loop_command.index('--node-sbatch') + 1:loop_command.index('--node-sbatch') + 3] == ['8', '4']
    assert loop_command[loop_command.index('--first-map-seconds') + 1] == '1200.0'
    assert loop_command.count('--optimizer-setting') == 2
    assert '-s' in loop_command[loop_command.index('--label') + 1]      # the MAP label records the settings
    assert '--accept-time-limit' not in loop_command
    assert run_ia_case.main(['--run-dir', str(case), '--stop', '5', '--ensemble-size', '8', '--accept-time-limit']) == 0
    assert len(commands) == 3 and '--accept-time-limit' in commands[2]


def test_budget_limited_map_is_final_only_when_accepted(tmp_path):
    import run_chain_laplace_loop as loop
    member = tmp_path / 'member-000'
    member.mkdir()
    (member / 'laplace.npz').write_bytes(b'')
    (member / 'result.json').write_text(json.dumps({'reason': 'time_limit', 'history': [{'objective': 3.}]}))
    assert loop.select_member(tmp_path, 1) is None
    assert loop.select_member(tmp_path, 1, accept_time_limit=True) == 0
    assert 'time_limit' in loop.final_reasons(True) and 'time_limit' not in loop.final_reasons()


def test_launcher_refuses_legacy_chain_when_global_prior_is_requested(tmp_path):
    import run_ia_case

    (tmp_path / 'study.json').write_text(json.dumps({'method': 'ia_field_dip_enif', 'ensemble_size': 250}))
    with pytest.raises(ValueError, match='different LWD prior'):
        run_ia_case.main(['--run-dir', str(tmp_path)])
