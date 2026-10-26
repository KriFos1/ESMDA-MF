import json
import pickle
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

import chain_laplace
from matern_prior import MaternPrior
import pilot_rml_index45 as rml
import run_DA_sequential as sequential
import sequential_matern_1d as driver


def run_chain(tmp_path, reference, name, *extra):
    output = tmp_path / name
    assert driver.main(['--reference', str(reference), '--output-dir', str(output), '--stop', '4',
                        '--ensemble-size', '12', '--mda-steps', '3', '--memory-ft', '20',
                        '--gate-threshold', '10', *extra]) == 0
    return output


def test_gate_probe_reverts_exports_and_stops(tmp_path, gate_environment):
    _, reference = gate_environment
    output = run_chain(tmp_path, reference, 'stop')
    history = json.loads((output / 'history.json').read_text())
    assert [entry['logging_index'] for entry in history] == [0, 1, 2]
    assert [entry['gate']['open'] for entry in history] == [False, False, True]
    assert history[2]['reverted'] and history[2]['gate']['post_member_median_misfit'] > 10
    assert history[2]['posterior']['rh_rmse_log10'] == history[2]['prior']['rh_rmse_log10']
    assert history[2]['steps'][-1]['member_1d_misfit_median'] > 10   # the probe trajectory is kept
    assert not (output / 'inversion_results_assim_2.pkl').exists()
    gate = json.loads((output / 'gate-002.json').read_text())
    assert gate['logging_index'] == 2 and gate['probe'] == history[2]['gate']
    problem = chain_laplace.load_problem(output / gate['problem'])
    assert problem['index'] == 2 and problem['prior_ensemble'].shape == (60, 12)
    assert problem['anisotropy'] == 'log_clipped' and problem['tools'] == [('6kHz', '83ft')]
    assert problem['well_X'].shape == (4, 1) and problem['ranges'].shape == (2, 2)
    with open(output / 'inversion_results_assim_1.pkl', 'rb') as stream:
        assert json.loads(json.dumps(pickle.load(stream)['gate']))['open'] is False


def test_gate_continue_carries_the_reverted_prior(tmp_path, gate_environment):
    _, reference = gate_environment
    output = run_chain(tmp_path, reference, 'continue', '--on-gate', 'continue')
    history = json.loads((output / 'history.json').read_text())
    assert [entry['reverted'] for entry in history] == [False, False, True, False]
    with open(output / 'inversion_results_assim_2.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    assert saved['reverted'] and saved['gate']['open']
    np.testing.assert_array_equal(np.asarray(saved['posterior_ensemble_params']).T, saved['prior_params'])
    assert not list(output.glob('laplace-problem-*.npz'))


def test_laplace_samples_match_the_exact_linear_gaussian_posterior():
    prior = MaternPrior((6, 8), (5., 5.), [[20., 25.], [30., 15.]], [1., .5], [.8, .5])
    n = len(prior.mean)
    rng = np.random.default_rng(3)
    jacobian = rng.normal(size=(5, n))
    variance = np.full(5, .04)
    z_map = rng.normal(size=n) * .1
    models, z, linear = chain_laplace.laplace_samples(prior, prior.mean, z_map, jacobian, variance, 6000, 11)
    whitened = prior.color(jacobian.T, transpose=True).T / np.sqrt(variance)[:, None]
    exact = np.linalg.inv(np.eye(n) + whitened.T @ whitened)
    empirical = np.cov(z)
    # Data directions are strongly shrunk (exact projected covariance ~ I_5): compare there,
    # and the total variance; a full 96 x 96 elementwise comparison is sampling-noise limited.
    projected_exact = whitened @ exact @ whitened.T
    projected = whitened @ empirical @ whitened.T
    assert np.linalg.norm(projected - projected_exact) / np.linalg.norm(projected_exact) < .1
    assert np.trace(projected_exact) < .05 * np.trace(whitened @ whitened.T)
    assert abs(np.trace(empirical) / np.trace(exact) - 1.) < .03
    assert np.abs(z.mean(axis=1)).max() < .1
    np.testing.assert_allclose(models - prior.mean[:, None] - prior.color(z_map)[:, None], prior.color(z), atol=1e-10)
    np.testing.assert_allclose(linear, jacobian @ prior.color(z), atol=1e-10)
    means, stds = chain_laplace.ensemble_moments(np.repeat(prior.mean[:, None], 3, axis=1) + 1e-3 * rng.normal(size=(n, 3)), n // 2)
    np.testing.assert_allclose(stds[0], .23) and np.testing.assert_allclose(stds[1], .1)
    np.testing.assert_allclose(means[0], prior.mean[:n // 2], atol=1e-2)


def test_prepare_laplace_and_sample_build_the_gate_checkpoint(tmp_path, gate_environment):
    _, reference = gate_environment
    chain = run_chain(tmp_path, reference, 'chain')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'study.json').write_text(json.dumps({
        'method': 'standalone_original_prior_rml', 'index': 45, 'count': 25, 'seed': 1,
        'ratio_max': 3., 'optimizer_method': rml.LM_OPTIMIZER,
        'optimizer': {'candidates': 4, 'initial_trust': .3, 'max_trust': 2., 'min_trust': 1e-3,
                      'max_iterations': 150, 'max_rounds': 4, 'max_seconds': 3300.,
                      'stagnation_tolerance': 1e-3, 'stagnation_iterations': 2,
                      'relative_gradient_tolerance': 1e-6},
        'resources': {'memory_gib': 48, 'walltime': '01:00:00'},
        'simulator': {'backend': 'local', 'cpus': 8, 'ranks': 2, 'concurrency': 4, 'physical_cores': True},
        'inputs_sha256': 'unused', 'reused_inputs_sha256': 'unused', 'cascade': {'old': True}}))
    run_dir = tmp_path / 'laplace-run'
    rml.prepare_laplace(SimpleNamespace(run_dir=run_dir, source=source, binary_dir=None, orientation=None,
                                        problem=chain / 'laplace-problem-002.npz'))
    study = json.loads((run_dir / 'study.json').read_text())
    assert 'orientation' not in study['simulator']
    override = tmp_path / 'laplace-run-bedding'
    rml.prepare_laplace(SimpleNamespace(run_dir=override, source=source, binary_dir='/builds/orient',
                                        orientation='bedding', problem=chain / 'laplace-problem-002.npz'))
    simulator = json.loads((override / 'study.json').read_text())['simulator']
    assert simulator['binary_dir'] == '/builds/orient' and simulator['orientation'] == 'bedding'
    assert simulator['cpus'] == 8
    assert study['method'] == 'chain_laplace_map' and study['count'] == 1 and study['index'] == 2
    assert study['prior']['type'] == 'matern_cellwise' and study['save_jacobian'] and 'cascade' not in study
    assert study['anisotropy'] == 'log_clipped' and '--array=0-0' in (run_dir / 'rml-array.sbatch').read_text()
    prior = rml.load_prior(study, run_dir)
    problem = chain_laplace.load_problem(chain / 'laplace-problem-002.npz')
    means, stds = chain_laplace.ensemble_moments(problem['prior_ensemble'], 30)
    np.testing.assert_allclose(prior.mean, np.concatenate(means))
    np.testing.assert_allclose(prior.std, np.concatenate(stds))
    assert prior.std.min() >= .1
    with np.load(run_dir / 'inputs.npz') as inputs:
        np.testing.assert_array_equal(inputs['prior_centers'][0], prior.mean)
        np.testing.assert_array_equal(inputs['targets'][0], problem['observations'])
        assert 'start_models' not in inputs.files and inputs['well_X'].shape == (4, 1)
    import tarfile
    rml.bundle(SimpleNamespace(run_dir=run_dir))
    with tarfile.open(run_dir / 'deployment.tar.gz') as archive:
        names = archive.getnames()
    assert 'run/prior-moments.npz' in names and 'run/inputs.npz' in names
    # Multi-start: same prior and data per member, each starting from a distinct carried member.
    starts = tmp_path / 'laplace-run-starts'
    rml.prepare_laplace(SimpleNamespace(run_dir=starts, source=source, binary_dir=None, orientation=None,
                                        problem=chain / 'laplace-problem-002.npz', starts=3, start_seed=5))
    with_mean = tmp_path / 'laplace-run-starts-mean'
    rml.prepare_laplace(SimpleNamespace(run_dir=with_mean, source=source, binary_dir=None, orientation=None,
                                        problem=chain / 'laplace-problem-002.npz', starts=3, start_seed=5,
                                        include_mean_start=True))
    meaned = json.loads((with_mean / 'study.json').read_text())
    assert meaned['count'] == 4 and meaned['starts']['labels'][0] == 'prior mean'
    with np.load(with_mean / 'inputs.npz') as inputs:
        np.testing.assert_array_equal(inputs['start_models'][0], inputs['prior_centers'][0])
        np.testing.assert_array_equal(inputs['start_models'][1:], problem['prior_ensemble'][:, meaned['starts']['members']].T)
    fast = tmp_path / 'laplace-run-fast'
    rml.prepare_laplace(SimpleNamespace(run_dir=fast, source=source, binary_dir=None, orientation=None,
                                        problem=chain / 'laplace-problem-002.npz',
                                        optimizer_setting=['stagnation_tolerance=0.01', 'stagnation_iterations=1',
                                                           'max_rounds=2']))
    optimizer = json.loads((fast / 'study.json').read_text())['optimizer']
    assert optimizer['stagnation_tolerance'] == .01 and optimizer['stagnation_iterations'] == 1
    assert optimizer['max_rounds'] == 2 and isinstance(optimizer['max_rounds'], int) and optimizer['candidates'] == 4
    with pytest.raises(ValueError, match='Unknown optimizer setting'):
        rml.prepare_laplace(SimpleNamespace(run_dir=tmp_path / 'bad', source=source, binary_dir=None,
                                            orientation=None, problem=chain / 'laplace-problem-002.npz',
                                            optimizer_setting=['tolerance=1']))
    node_run = tmp_path / 'laplace-run-node'
    rml.prepare_laplace(SimpleNamespace(run_dir=node_run, source=source, binary_dir='/data1/orient', orientation=None,
                                        problem=chain / 'laplace-problem-002.npz', node_solver=(3, 4),
                                        hdfql='/data1/orient'))
    node_study = json.loads((node_run / 'study.json').read_text())
    assert node_study['execution'] == 'node' and node_study['clean_jobs']
    assert {key: node_study['simulator'][key] for key in ('ranks', 'cpus', 'mkl_threads', 'concurrency',
                                                          'srun_steps', 'physical_cores', 'hdfql', 'binary_dir')} == {
        'ranks': 1, 'cpus': 3, 'mkl_threads': 3, 'concurrency': 4, 'srun_steps': False, 'physical_cores': False,
        'hdfql': '/data1/orient', 'binary_dir': '/data1/orient'}
    assert 'execution' not in study and not study.get('clean_jobs')
    multi = json.loads((starts / 'study.json').read_text())
    members = multi['starts']['members']
    assert multi['count'] == 3 and len(set(members)) == 3 and '--array=0-2' in (starts / 'rml-array.sbatch').read_text()
    with np.load(starts / 'inputs.npz') as inputs:
        np.testing.assert_array_equal(inputs['prior_centers'], np.tile(prior.mean, (3, 1)))
        np.testing.assert_array_equal(inputs['targets'], np.tile(problem['observations'], (3, 1)))
        np.testing.assert_array_equal(inputs['start_models'], problem['prior_ensemble'][:, members].T)
    # Stationary Matern prior of the chain, explicit start models with labels.
    starts_file = tmp_path / 'starts.npz'
    np.savez(starts_file, models=problem['prior_ensemble'][:, :2].T, labels=np.array(['first', 'second']))
    matern_run = tmp_path / 'laplace-run-matern'
    rml.prepare_laplace(SimpleNamespace(run_dir=matern_run, source=source, binary_dir=None, orientation=None,
                                        problem=chain / 'laplace-problem-002.npz', prior='matern',
                                        start_models=starts_file, likelihood_weight=5.))
    matern_study = json.loads((matern_run / 'study.json').read_text())
    chain_prior = json.loads((chain / 'study.json').read_text())['prior']
    assert matern_study['prior']['stationary'] == {'means': chain_prior['means'], 'stds': chain_prior['stds']}
    assert matern_study['count'] == 2 and matern_study['starts']['labels'] == ['first', 'second']
    stationary = rml.load_prior(matern_study, matern_run)
    np.testing.assert_allclose(stationary.mean, np.repeat(chain_prior['means'], 30))
    np.testing.assert_allclose(stationary.std, np.repeat(chain_prior['stds'], 30))
    with np.load(matern_run / 'inputs.npz') as inputs:
        np.testing.assert_array_equal(inputs['start_models'], problem['prior_ensemble'][:, :2].T)
        np.testing.assert_array_equal(inputs['prior_centers'][1], stationary.mean)
        np.testing.assert_allclose(inputs['variance'], problem['variance'] / 5.)
    assert matern_study['likelihood_weight'] == 5.
    with pytest.raises(ValueError, match='either'):
        rml.prepare_laplace(SimpleNamespace(run_dir=tmp_path / 'both', source=source, binary_dir=None,
                                            orientation=None, problem=chain / 'laplace-problem-002.npz',
                                            starts=2, start_models=starts_file))
    # Fake MAP result: the prior mean itself, with a linear response of the bit column.
    member = run_dir / 'member-000'
    member.mkdir()
    nd = len(problem['observations'])
    jacobian = np.zeros((nd, 60))
    jacobian[:, 12:18] = 1. / 6.
    prediction = np.full(nd, float(prior.mean[12:18].mean()))
    np.savez(member / 'fit.npz', model=prior.mean, z=np.zeros(60), prediction=prediction)
    np.savez(member / 'laplace.npz', model=prior.mean, prediction=prediction, jacobian=jacobian)
    (member / 'result.json').write_text(json.dumps({'reason': 'objective_stagnation', 'converged': True,
                                                    'accepted_iterations': 3, 'seconds_this_attempt': 9.}))
    assert chain_laplace.main(['sample', '--chain', str(chain), '--map', str(run_dir),
                               '--count', '40', '--seed', '1']) == 0
    with open(chain / 'inversion_results_assim_2.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    assert saved['selected_fidelity'] == '2D' and saved['final_stage'] == 'laplace' and saved['reverted']
    assert np.shape(saved['posterior_ensemble_params']) == (40, 60)
    assert saved['misfit']['before_final'] > 10 and saved['laplace']['count'] == 40
    samples = np.asarray(saved['posterior_ensemble_params']).T
    assert samples[12:18].std(axis=1).mean() < prior.std[12:18].mean()   # data tightened the bit column
    assert samples[:30].max() <= np.log(40.) + 1e-12
    history = json.loads((chain / 'history.json').read_text())
    assert history[-1]['stage'] == 'laplace_2d' and history[-1]['logging_index'] == 2
    assert (chain / 'laplace-002.json').exists()


def test_loop_helpers_find_pending_gates_and_parse_sacct(tmp_path):
    import run_chain_laplace_loop as loop
    chain = tmp_path / 'chain'
    chain.mkdir()
    for index in (0, 1, 2):
        (chain / f'inversion_results_assim_{index}.pkl').write_bytes(b'')
    assert loop.pending_gate(chain) is None
    (chain / 'gate-002.json').write_text('{}')        # already sampled: checkpoint 2 exists
    assert loop.pending_gate(chain) is None
    (chain / 'gate-003.json').write_text('{}')
    assert loop.pending_gate(chain) == 3 and loop.checkpoint_indices(chain) == [0, 1, 2]
    text = '1760507_0|RUNNING\n1760507_0.batch|RUNNING\n1760507_0.extern|COMPLETED\n'
    assert loop.sacct_state(text, '1760507') == 'RUNNING'
    assert loop.sacct_state(text.replace('RUNNING', 'COMPLETED'), '1760507') == 'COMPLETED'
    assert loop.sacct_state('', '1760507') is None
    assert loop.sacct_state('1760507_0|CANCELLED by 123\n', '1760507') == 'CANCELLED'
    study = {'ensemble_size': 250, 'seed': 2031, 'mda_steps': 10, 'tempering': 'linear-inflation',
             'anisotropy': {'parameterization': 'log_clipped', 'state_moments': [.5, .5]},
             'relaxation': {'memory_ft': 50.}, 'gate': {'threshold': 10.}}
    arguments = loop.driver_arguments(study, 51)
    assert arguments[:2] == ['--stop', '51'] and '--resume' in arguments
    assert arguments[arguments.index('--memory-ft') + 1] == '50.0'
    assert arguments[arguments.index('--gate-threshold') + 1] == '10.0'
    assert '--memory-ft' not in loop.driver_arguments({**study, 'relaxation': 'none', 'gate': 'none'}, 51)
    # The resumed driver keeps the chain's own reference (its run.json supplies the data).
    assert '--reference' not in arguments
    referenced = loop.driver_arguments({**study, 'reference': '/runs/ref'}, 51)
    assert referenced[referenced.index('--reference') + 1] == '/runs/ref'


def test_carry_fallback_writes_the_gate_point_with_its_prior(tmp_path, gate_environment):
    _, reference = gate_environment
    chain = run_chain(tmp_path, reference, 'fallback')
    assert chain_laplace.main(['carry', '--chain', str(chain), '--reason', 'test failure']) == 0
    with open(chain / 'inversion_results_assim_2.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    problem = chain_laplace.load_problem(chain / 'laplace-problem-002.npz')
    np.testing.assert_array_equal(saved['post_mda_params'], problem['prior_ensemble'])
    assert saved['final_stage'] == 'carried_prior_fallback' and saved['selected_fidelity'] == '1D'
    assert saved['fallback_reason'] == 'test failure' and saved['reverted']
    history = json.loads((chain / 'history.json').read_text())
    assert history[-1]['stage'] == 'carried_prior_fallback'
    with pytest.raises(FileExistsError):
        chain_laplace.main(['carry', '--chain', str(chain)])
    import run_chain_laplace_loop as loop
    assert loop.pending_gate(chain) is None     # the chain can resume past the fallback point


def test_retry_repeats_transient_remote_failures(monkeypatch):
    import subprocess
    import run_chain_laplace_loop as loop
    monkeypatch.setattr(loop.time, 'sleep', lambda seconds: None)
    calls = {'n': 0}

    def flaky():
        calls['n'] += 1
        if calls['n'] < 3:
            raise subprocess.CalledProcessError(255, ['ssh'])
        return 'ok'
    assert loop.retry(flaky, attempts=4, delay=0.) == 'ok' and calls['n'] == 3
    with pytest.raises(subprocess.CalledProcessError):
        loop.retry(lambda: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ['x'])), attempts=2, delay=0.)


def test_loop_pause_file_stops_before_any_work(tmp_path, monkeypatch):
    import run_chain_laplace_loop as loop
    chain = tmp_path / 'chain'
    chain.mkdir()
    (chain / 'study.json').write_text('{}')
    (chain / 'PAUSE').write_text('')
    monkeypatch.setattr(loop, 'run', lambda *args, **kwargs: pytest.fail('nothing may run while paused'))
    assert loop.main(['--chain', str(chain), '--stop', '10']) == 0


@pytest.mark.parametrize('remote_state', ['in_flight', 'finished'])
def test_map_stage_adopts_a_job_or_result_left_by_a_previous_controller(tmp_path, monkeypatch, remote_state):
    import run_chain_laplace_loop as loop
    monkeypatch.setattr(loop, 'ROOT', tmp_path)
    member = tmp_path / 'runs' / 'chain-laplace-idx007-test' / 'member-000'
    member.mkdir(parents=True)                       # prepared and deployed earlier
    (member.parent / 'study.json').write_text(json.dumps({'count': 1}))
    commands, waited = [], []

    def arrive():
        (member / 'result.json').write_text(json.dumps({'reason': 'objective_stagnation',
                                                        'history': [{'objective': 3.}]}))
        (member / 'laplace.npz').write_bytes(b'')

    def fake_run(command, capture=False, env=None):
        commands.append(' '.join(str(part) for part in command))
        if command[0] == 'rsync' and (remote_state == 'finished' or waited):
            arrive()

    def fake_ssh(host, command, capture=False):
        commands.append(command)
        assert 'sbatch' not in command
        return '555|/remote/chain-laplace-idx007-test/run/\n999|/elsewhere\n' if 'squeue' in command else ''

    monkeypatch.setattr(loop, 'run', fake_run)
    monkeypatch.setattr(loop, 'ssh', fake_ssh)
    monkeypatch.setattr(loop, 'wait_for_array', lambda host, job, members, poll: waited.append(job))
    args = SimpleNamespace(label='test', hpc_root='/remote', host='HPC', source=tmp_path, poll=0.,
                           max_submissions=3, count=5, seed_base=0, hpc_python='python')
    loop.map_stage(args, tmp_path / 'chain', 7)
    assert waited == (['555'] if remote_state == 'in_flight' else [])
    assert any('chain_laplace.py sample' in command for command in commands)
    assert not any('prepare-laplace' in command for command in commands)
    assert loop.job_in_directory('12|/a/run\n13|/b/run/\n', '/b/run') == '13'
    assert loop.job_in_directory('', '/b/run') is None


def test_fresh_map_stage_passes_the_build_and_orientation_to_prepare(tmp_path, monkeypatch):
    import run_chain_laplace_loop as loop
    monkeypatch.setattr(loop, 'ROOT', tmp_path)
    member = tmp_path / 'runs' / 'chain-laplace-idx007-test' / 'member-000'
    commands = []

    def fake_run(command, capture=False, env=None):
        commands.append([str(part) for part in command])
        if 'prepare-laplace' in commands[-1]:
            member.mkdir(parents=True)
            (member.parent / 'study.json').write_text(json.dumps({'count': 1}))
        if command[0] == 'rsync':
            (member / 'result.json').write_text(json.dumps({'reason': 'objective_stagnation',
                                                            'history': [{'objective': 3.}]}))
            (member / 'laplace.npz').write_bytes(b'')

    monkeypatch.setattr(loop, 'run', fake_run)
    monkeypatch.setattr(loop, 'ssh', lambda host, command, capture=False: '555\n' if 'sbatch' in command else '')
    monkeypatch.setattr(loop, 'wait_for_array', lambda host, job, members, poll: None)
    args = SimpleNamespace(label='test', hpc_root='/remote', host='HPC', source=tmp_path, poll=0.,
                           max_submissions=3, count=5, seed_base=0, hpc_python='python',
                           binary_dir='/builds/orient', orientation='bedding')
    loop.map_stage(args, tmp_path / 'chain', 7)
    prepare = next(command for command in commands if 'prepare-laplace' in command)
    assert prepare[prepare.index('--binary-dir') + 1] == '/builds/orient'
    assert prepare[prepare.index('--orientation') + 1] == 'bedding'
    assert any('chain_laplace.py' in ' '.join(command) and 'sample' in command for command in commands)
    commands.clear()
    member.parent.rename(tmp_path / 'previous')
    loop.map_stage(SimpleNamespace(**{**vars(args), 'binary_dir': None, 'orientation': None}),
                   tmp_path / 'chain', 7)
    prepare = next(command for command in commands if 'prepare-laplace' in command)
    assert '--binary-dir' not in prepare and '--orientation' not in prepare and '--starts' not in prepare


def test_multi_start_map_stage_resamples_the_lowest_objective(tmp_path, monkeypatch):
    import run_chain_laplace_loop as loop
    monkeypatch.setattr(loop, 'ROOT', tmp_path)
    run_dir = tmp_path / 'runs' / 'chain-laplace-idx007-test'
    commands, submitted = [], []
    objectives = {0: 40., 1: 12., 2: 25.}

    def fake_run(command, capture=False, env=None):
        commands.append([str(part) for part in command])
        if 'prepare-laplace' in commands[-1]:
            run_dir.mkdir(parents=True)
            (run_dir / 'study.json').write_text(json.dumps({'count': 3}))
        if command[0] == 'rsync':
            for member, objective in objectives.items():
                if member == 2 and len(submitted) < 2:
                    continue                                 # member 2 needs a second submission
                path = run_dir / f'member-{member:03d}'
                path.mkdir(exist_ok=True)
                (path / 'result.json').write_text(json.dumps({'reason': 'objective_stagnation',
                                                              'history': [{'objective': objective}]}))
                (path / 'laplace.npz').write_bytes(b'')

    def fake_ssh(host, command, capture=False):
        if 'sbatch' in command:
            submitted.append(command)
            return f'{555 + len(submitted)}\n'
        return ''

    monkeypatch.setattr(loop, 'run', fake_run)
    monkeypatch.setattr(loop, 'ssh', fake_ssh)
    monkeypatch.setattr(loop, 'wait_for_array', lambda host, job, members, poll: None)
    args = SimpleNamespace(label='test', hpc_root='/remote', host='HPC', source=tmp_path, poll=0., max_submissions=3,
                           count=5, seed_base=100, hpc_python='python', binary_dir=None, orientation=None,
                           map_starts=2)
    loop.map_stage(args, tmp_path / 'chain', 7)
    prepare = next(command for command in commands if 'prepare-laplace' in command)
    assert prepare[prepare.index('--starts') + 1] == '2' and '--include-mean-start' in prepare
    assert prepare[prepare.index('--start-seed') + 1] == '107'
    assert '--array=0,1,2' in submitted[0] and '--array=2' in submitted[1]
    assert '--optimizer-setting' not in prepare
    assert json.loads((run_dir / 'selected-member.json').read_text())['member'] == 1
    assert chain_laplace.selected_member(run_dir) == 1
    assert loop.array_states('9_0|COMPLETED\n9_0.batch|COMPLETED\n9_[1-2]|PENDING\n', '9') == {0: 'COMPLETED'}


def test_phase_alignment_wraps_residuals_and_keeps_other_channels():
    from esmda_core import align_phases, phase_channels
    from esmda_remote2d import UDAR_COMPONENTS
    mask = np.ones(16, bool)
    mask[[0, 1, 8, 9]] = False                       # ultra-shallow USDP/USDA of two tools
    phase = phase_channels(mask, UDAR_COMPONENTS)
    assert phase.tolist() == [True, False, True, False, True, False] * 2
    observations = np.array([175., 3., -170., 1., 10., 2.] * 2)
    predictions = np.array([-178., 3.5, 175., 1.2, 12., 2.1] * 2)
    aligned = align_phases(predictions, observations, phase)
    np.testing.assert_allclose(aligned, [182., 3.5, -185., 1.2, 12., 2.1] * 2)
    residual = observations - aligned
    assert np.all(np.abs(residual[phase]) <= 180.)
    members = np.column_stack([predictions, predictions + 720. * phase])
    np.testing.assert_allclose(align_phases(members, observations, phase), np.column_stack([aligned, aligned]))
    np.testing.assert_allclose(align_phases(members.T, observations, phase, axis=1), np.vstack([aligned, aligned]))
    # Predictions already on the observed branch come back bit for bit.
    near = observations + np.linspace(-170., 170., observations.size) / 7.
    assert np.array_equal(align_phases(near, observations, phase), near)
    with pytest.raises(ValueError, match='whole tools'):
        phase_channels(np.ones(6, bool), UDAR_COMPONENTS)


def test_fork_chain_copies_state_through_an_index(tmp_path, gate_environment):
    _, reference = gate_environment
    chain = run_chain(tmp_path, reference, 'source', '--on-gate', 'continue')
    fork = driver.fork_chain(chain, tmp_path / 'fork', 1, 'restart 2D segment')
    assert sorted(p.name for p in fork.glob('inversion_results_assim_*.pkl')) == [
        'inversion_results_assim_0.pkl', 'inversion_results_assim_1.pkl']
    history = json.loads((fork / 'history.json').read_text())
    assert [entry['logging_index'] for entry in history] == [0, 1]
    study = json.loads((fork / 'study.json').read_text())
    assert study['forked_from'] == {'chain': str(chain.resolve()), 'last_index': 1, 'note': 'restart 2D segment'}
    assert all(entry['from_index'] <= 1 for entry in study.get('resumes', []))
    with pytest.raises(FileExistsError):
        driver.fork_chain(chain, fork, 1, 'again')
    with pytest.raises(FileNotFoundError):
        driver.fork_chain(chain, tmp_path / 'too-far', 9, 'missing')
    # The fork resumes at index 2 with the probe re-run (and its phase-wrap count recorded).
    assert driver.main(['--reference', str(reference), '--output-dir', str(fork), '--stop', '3',
                        '--ensemble-size', '12', '--mda-steps', '3', '--memory-ft', '20',
                        '--gate-threshold', '10', '--on-gate', 'continue', '--resume']) == 0
    history = json.loads((fork / 'history.json').read_text())
    assert [entry['logging_index'] for entry in history] == [0, 1, 2]
    assert history[-1]['phase_wraps'] == 0


def single_point(tmp_path, reference, name):
    output = tmp_path / name
    assert driver.main(['--reference', str(reference), '--output-dir', str(output), '--start', '1',
                        '--stop', '2', '--ensemble-size', '12', '--mda-steps', '5',
                        '--tempering', 'equal-energy', '--likelihood-weight', '0.5']) == 0
    return output


def test_likelihood_weight_tempers_every_step(tmp_path, gate_environment):
    _, reference = gate_environment
    output = single_point(tmp_path, reference, 'weighted')
    study = json.loads((output / 'study.json').read_text())
    assert study['likelihood_weight'] == .5 and study['covariance_inflation'] == [10.] * 5
    history = json.loads((output / 'history.json').read_text())
    assert [entry['logging_index'] for entry in history] == [1]
    assert [step['alpha'] for step in history[0]['steps'][1:]] == [10.] * 5
    with open(output / 'inversion_results_assim_1.pkl', 'rb') as stream:
        assert pickle.load(stream)['mda_weight'] == .5
    for weight in ('0', '1.5', 'nan'):
        with pytest.raises(SystemExit):
            driver.parse_args(['--output-dir', str(tmp_path / 'x'), '--likelihood-weight', weight])


def test_handoff_check_separates_simulator_misalignment_from_lateral_structure(
        tmp_path, gate_environment, monkeypatch):
    import handoff_misalignment_check as handoff
    _, reference = gate_environment
    run = single_point(tmp_path, reference, 'single')
    with open(run / 'inversion_results_assim_1.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    best = int(np.argmin(saved['posterior_misfit']))
    monkeypatch.setattr(sequential, 'remote_2d', getattr(sequential, 'remote_2d', None), raising=False)
    batches = []

    class Adapter:
        jobs_dir = None

        def run_pred_multi(self, param_dicts, index, shift, well):
            batches.append((len(param_dicts), index, dict(shift)))
            rows = []
            for number, parameters in enumerate(param_dicts):
                # 2D = 1D + 0.5 on the same layered medium; the full field also gains a
                # 360-degree turn on its phases, which the check must undo.
                row = np.full(8, float(np.mean(parameters['rh']))) + .5
                if number == 0:
                    row[[0, 2, 4, 6]] += 360.
                rows.append(row)
            return np.array(rows)

    monkeypatch.setattr(handoff, 'make_remote2d', lambda *args: Adapter())
    output = tmp_path / 'handoff'
    assert handoff.main(['--run-dir', str(run), '--index', '1', '--output-dir', str(output)]) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['member'] == best and report['one_d_matches_checkpoint']
    assert batches == [(2, 1, saved['shift_ft'])]
    assert report['data_misfit']['1d'] == pytest.approx(saved['posterior_misfit'][best])
    assert report['simulator_misalignment']['chi2_per_channel'] == pytest.approx(1.)   # (0.5 / 0.5)^2
    nx, nz = 5, 6
    field = saved['posterior_params'][best].reshape(2, nx, nz)
    lateral = (field[0].mean() - field[0, nx // 2].mean()) ** 2 / .25
    assert report['lateral_structure']['chi2_per_channel'] == pytest.approx(lateral)
    assert sum(report['simulator_misalignment']['share_by_component'].values()) == pytest.approx(1.)
    with np.load(output / 'predictions.npz') as saved_predictions:
        invariant = saved_predictions['field_2d_invariant'].reshape(2, nx, nz)
        assert np.array_equal(invariant, np.repeat(field[:, nx // 2:nx // 2 + 1], nx, axis=1))
        assert np.all(np.abs(saved_predictions['prediction_2d_full'] - saved['observations']) <= 180.)
    with pytest.raises(FileExistsError):
        handoff.main(['--run-dir', str(run), '--index', '1', '--output-dir', str(output)])


def test_handoff_check_includes_the_members_prior_field(tmp_path, gate_environment, monkeypatch):
    import handoff_misalignment_check as handoff
    _, reference = gate_environment
    run = single_point(tmp_path, reference, 'single')
    with open(run / 'inversion_results_assim_1.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    best = int(np.argmin(saved['posterior_misfit']))
    monkeypatch.setattr(sequential, 'remote_2d', getattr(sequential, 'remote_2d', None), raising=False)
    fields = []

    class Adapter:
        jobs_dir = None

        def run_pred_multi(self, param_dicts, index, shift, well):
            fields.extend(np.asarray(parameters['rh']) for parameters in param_dicts)
            return np.array([np.full(8, float(np.mean(parameters['rh']))) + .5 for parameters in param_dicts])

    monkeypatch.setattr(handoff, 'make_remote2d', lambda *args: Adapter())
    output = tmp_path / 'handoff-prior'
    assert handoff.main(['--run-dir', str(run), '--index', '1', '--output-dir', str(output),
                         '--include-prior']) == 0
    report = json.loads((output / 'report.json').read_text())
    assert len(fields) == 4
    nx, nz = 5, 6
    prior = saved['prior_params'][:, best].reshape(2, nx, nz)
    observations = saved['observations']
    expected_1d = np.mean((observations - prior[0, nx // 2].mean()) ** 2 / .25)
    expected_2d = np.mean((observations - prior[0].mean() - .5) ** 2 / .25)
    assert report['data_misfit']['1d_prior'] == pytest.approx(expected_1d)
    assert report['data_misfit']['2d_prior_full'] == pytest.approx(expected_2d)
    assert report['prior_simulator_misalignment']['chi2_per_channel'] == pytest.approx(1.)
    update = (saved['posterior_params'][best].reshape(2, nx, nz)[0].mean() - prior[0].mean()) ** 2 / .25
    assert report['update_effect_2d_full']['chi2_per_channel'] == pytest.approx(update)
    with np.load(output / 'predictions.npz') as saved_predictions:
        assert np.array_equal(saved_predictions['field_2d_prior_full'], saved['prior_params'][:, best])


def run_cascade(tmp_path, reference, name, *extra, stop='4'):
    output = tmp_path / name
    assert driver.main(['--reference', str(reference), '--output-dir', str(output), '--stop', stop,
                        '--ensemble-size', '12', '--mda-steps', '3', '--tempering', 'equal-energy',
                        '--likelihood-weight', '0.5', '--memory-ft', '20', '--cascade-threshold', '10',
                        '--compare-revert', 'first', *extra]) == 0
    return output


def test_cascade_finishes_easy_points_in_1d_and_sends_hard_ones_to_2d_with_the_update(
        tmp_path, gate_environment):
    _, reference = gate_environment
    chain = run_cascade(tmp_path, reference, 'cascade')
    history = json.loads((chain / 'history.json').read_text())
    assert [entry['cascade']['decision'] for entry in history] == ['1d', '1d', '2d']
    assert [step['alpha'] for step in history[0]['steps'][1:]] == [6.] * 6   # both halves, 3 steps each
    assert [step['alpha'] for step in history[2]['steps'][1:]] == [6.] * 3   # first half only
    assert not any(entry['reverted'] for entry in history)
    assert history[2]['cascade']['best_member_misfit'] > 10 > history[0]['cascade']['best_member_misfit']
    with open(chain / 'inversion_results_assim_1.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    assert saved['mda_weight'] == 1. and saved['covariance_inflation'] == [6.] * 6
    assert saved['cascade']['decision'] == '1d' and not saved['reverted']
    study = json.loads((chain / 'study.json').read_text())
    assert study['cascade']['threshold'] == 10 and study['cascade']['completion_inflation'] == [6.] * 3
    gate = json.loads((chain / 'gate-002.json').read_text())
    assert gate['cascade']['decision'] == '2d' and gate['revert_problem'] == 'laplace-problem-002-revert.npz'
    keep = chain_laplace.load_problem(chain / gate['problem'])
    revert = chain_laplace.load_problem(chain / gate['revert_problem'])
    assert keep['likelihood_weight'] == .5 and keep['prior_type'] == 'ensemble_precision'
    assert revert['likelihood_weight'] == 1. and revert['prior_type'] == 'ensemble_precision'
    assert not keep['probe']['reverted'] and revert['probe']['reverted']
    # The kept ensemble carries the 1D update toward the high observation; the reverted one does not.
    bit = slice(2 * 6, 3 * 6)
    assert keep['prior_ensemble'][bit].mean() > revert['prior_ensemble'][bit].mean()
    assert not (chain / 'inversion_results_assim_2.pkl').exists()
    for flags in (['--gate-threshold', '5'], ['--on-gate', 'continue'], ['--likelihood-weight', '1']):
        with pytest.raises(SystemExit):
            driver.parse_args(['--output-dir', str(tmp_path / 'x'), '--cascade-threshold', '10',
                               '--likelihood-weight', '0.5', *flags])
    with pytest.raises(SystemExit):
        driver.parse_args(['--output-dir', str(tmp_path / 'x'), '--compare-revert', 'first'])


def test_cascade_exports_the_reverted_problem_only_at_the_first_gate(tmp_path, gate_environment):
    _, reference = gate_environment
    chain = run_cascade(tmp_path, reference, 'cascade-first')
    assert chain_laplace.main(['carry', '--chain', str(chain), '--reason', 'test']) == 0
    (chain / 'gate-002.json').rename(chain / 'gate-002.done')     # keep only one pending-gate record
    (chain / 'gate-000.json').write_text('{}')                     # an earlier gate already happened
    assert driver.main(['--reference', str(reference), '--output-dir', str(chain), '--stop', '4',
                        '--ensemble-size', '12', '--mda-steps', '3', '--tempering', 'equal-energy',
                        '--likelihood-weight', '0.5', '--memory-ft', '20', '--cascade-threshold', '1e-6',
                        '--compare-revert', 'first', '--resume']) == 0
    gate = json.loads((chain / 'gate-003.json').read_text())
    assert gate['cascade']['decision'] == '2d' and gate['revert_problem'] is None
    assert not (chain / 'laplace-problem-003-revert.npz').exists()


def fake_map(run_dir, prior, problem):
    """A finished MAP at the prior mean with a linear response of the bit column."""
    member = run_dir / 'member-000'
    member.mkdir()
    nd = len(problem['observations'])
    jacobian = np.zeros((nd, 60))
    jacobian[:, 12:18] = 1. / 6.
    prediction = np.full(nd, float(prior.mean[12:18].mean()))
    np.savez(member / 'fit.npz', model=prior.mean, z=np.zeros(60), prediction=prediction)
    np.savez(member / 'laplace.npz', model=prior.mean, prediction=prediction, jacobian=jacobian)
    (member / 'result.json').write_text(json.dumps({'reason': 'objective_stagnation', 'converged': True,
                                                    'accepted_iterations': 3, 'seconds_this_attempt': 9.}))


def test_cascade_2d_stage_uses_the_ensemble_precision_and_the_remaining_share(
        tmp_path, gate_environment, monkeypatch):
    from sparse_precision import EnsemblePrecisionPrior
    _, reference = gate_environment
    chain = run_cascade(tmp_path, reference, 'cascade-2d')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'study.json').write_text(json.dumps({
        'method': 'standalone_original_prior_rml', 'index': 45, 'count': 25, 'seed': 1, 'ratio_max': 3.,
        'optimizer_method': rml.LM_OPTIMIZER, 'optimizer': {'candidates': 4},
        'resources': {'memory_gib': 48, 'walltime': '01:00:00'},
        'simulator': {'backend': 'local', 'cpus': 8, 'ranks': 2, 'concurrency': 4, 'physical_cores': True},
        'inputs_sha256': 'unused'}))
    keep_problem = chain / 'laplace-problem-002.npz'
    run_dir = tmp_path / 'keep-run'
    rml.prepare_laplace(SimpleNamespace(run_dir=run_dir, source=source, binary_dir=None, orientation='bedding',
                                        problem=keep_problem))
    study = json.loads((run_dir / 'study.json').read_text())
    assert study['prior']['type'] == 'ensemble_precision' and study['likelihood_weight'] == .5
    assert study['prior']['sha256'] == chain_laplace.digest(run_dir / 'prior-precision.npz')
    problem = chain_laplace.load_problem(keep_problem)
    prior = rml.load_prior(study, run_dir)
    assert isinstance(prior, EnsemblePrecisionPrior)
    np.testing.assert_allclose(prior.mean, problem['prior_ensemble'].mean(axis=1))
    assert prior.std[:30].min() >= .23 - 1e-12
    with np.load(run_dir / 'inputs.npz') as inputs:
        np.testing.assert_allclose(inputs['variance'], 2. * problem['variance'])
        np.testing.assert_array_equal(inputs['prior_centers'][0], prior.mean)
        assert 'start_models' not in inputs.files              # start: the ensemble mean
    import tarfile
    rml.bundle(SimpleNamespace(run_dir=run_dir))
    with tarfile.open(run_dir / 'deployment.tar.gz') as archive:
        names = archive.getnames()
    assert 'run/prior-precision.npz' in names and 'code/sparse_precision.py' in names
    fake_map(run_dir, prior, problem)
    variances = []
    original = chain_laplace.laplace_samples

    def spy(prior, center, z_map, jacobian, variance, count, seed):
        variances.append(np.asarray(variance).copy())
        return original(prior, center, z_map, jacobian, variance, count, seed)
    monkeypatch.setattr(chain_laplace, 'laplace_samples', spy)
    assert chain_laplace.main(['sample', '--chain', str(chain), '--map', str(run_dir),
                               '--count', '40', '--seed', '1']) == 0
    np.testing.assert_allclose(variances[-1], 2. * problem['variance'])
    with open(chain / 'inversion_results_assim_2.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    assert not saved['reverted'] and saved['mda_weight'] == .5 and saved['lower_fidelity_weight'] == .5
    assert saved['two_d_prior'] == 'ensemble_precision' and saved['cascade']['decision'] == '2d'
    assert saved['laplace']['likelihood_weight'] == .5
    # Side-by-side: the reverted problem, full energy, resampled outside the chain.
    revert_run = tmp_path / 'revert-run'
    revert_problem = chain / 'laplace-problem-002-revert.npz'
    rml.prepare_laplace(SimpleNamespace(run_dir=revert_run, source=source, binary_dir=None,
                                        orientation='bedding', problem=revert_problem))
    revert_prior = rml.load_prior(json.loads((revert_run / 'study.json').read_text()), revert_run)
    fake_map(revert_run, revert_prior, chain_laplace.load_problem(revert_problem))
    before = sorted(path.name for path in chain.iterdir())
    output = tmp_path / 'revert-compare'
    assert chain_laplace.main(['compare', '--chain', str(chain), '--problem', str(revert_problem),
                               '--map', str(revert_run), '--output', str(output), '--count', '40']) == 0
    np.testing.assert_allclose(variances[-1], problem['variance'])
    assert sorted(path.name for path in chain.iterdir()) == before
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['posterior']['likelihood_weight'] == 1. and summary['probe']['reverted']
    with np.load(output / 'samples.npz') as samples:
        assert samples['models'].shape == (60, 40)


def test_controller_resumes_a_cascade_chain_with_its_settings(tmp_path):
    import run_chain_laplace_loop as loop
    study = {'ensemble_size': 250, 'seed': 2031, 'mda_steps': 5, 'tempering': 'equal-energy',
             'anisotropy': {'parameterization': 'log_clipped', 'state_moments': [.5, .5]},
             'relaxation': {'memory_ft': 50.}, 'gate': 'none', 'likelihood_weight': .5,
             'cascade': {'threshold': 100., 'compare_revert': 'first'}}
    arguments = loop.driver_arguments(study, 51)
    assert arguments[arguments.index('--likelihood-weight') + 1] == '0.5'
    assert arguments[arguments.index('--cascade-threshold') + 1] == '100.0'
    assert arguments[arguments.index('--compare-revert') + 1] == 'first'
    assert '--gate-threshold' not in arguments and '--resume' in arguments
    (tmp_path / 'study.json').write_text('{}')
    parsed = driver.parse_args(['--output-dir', str(tmp_path), *arguments])
    assert parsed.cascade_threshold == 100. and parsed.likelihood_weight == .5 and parsed.compare_revert == 'first'


def test_controller_runs_the_side_by_side_problem_before_the_gate_point(tmp_path, monkeypatch):
    import run_chain_laplace_loop as loop
    chain = tmp_path / 'chain'
    chain.mkdir()
    (chain / 'study.json').write_text(json.dumps({'gate': 'none'}))
    (chain / 'gate-003.json').write_text(json.dumps({'revert_problem': 'laplace-problem-003-revert.npz'}))
    calls = []

    def fake_stage(args, chain_dir, index, problem=None, suffix='', compare=False):
        calls.append((index, problem, suffix, compare))
        if not compare:
            (chain_dir / f'inversion_results_assim_{index}.pkl').write_bytes(b'')
            (chain_dir / 'PAUSE').write_text('')
    monkeypatch.setattr(loop, 'map_stage', fake_stage)
    monkeypatch.setattr(loop, 'ROOT', tmp_path)
    assert loop.main(['--chain', str(chain), '--stop', '10']) == 0
    assert calls == [(3, 'laplace-problem-003-revert.npz', '-revert', True), (3, None, '', False)]


def test_lateral_weights_follow_the_rules():
    np.testing.assert_array_equal(driver.lateral_weights(7, 5., 'column', column_halfwidth=1), [0, 0, 1, 1, 1, 0, 0])
    footprint = driver.lateral_weights(33, 5., 'footprint')        # bit 16; 40 ft full, zero at 60 ft
    assert np.all(footprint[16 - 8:16 + 9] == 1.) and np.all(footprint[:4] == 0.) and np.all(footprint[-4:] == 0.)
    assert 0 < footprint[16 - 10] < 1 and footprint[16 - 10] == footprint[16 + 10]
    assert np.all(np.diff(footprint[:17]) >= 0)
    np.testing.assert_array_equal(driver.lateral_weights(5, 5., 'full'), np.ones(5))
    with pytest.raises(SystemExit):
        driver.parse_args(['--output-dir', 'x', '--footprint-ft', '60', '--footprint-taper-ft', '40'])


def test_lateral_update_modes_share_the_bit_column_and_shape_the_flanks(tmp_path, gate_environment):
    _, reference = gate_environment
    nx, nz = 5, 6
    runs = {}
    for mode, extra in (('full', []), ('column', ['--column-halfwidth', '1']),
                        ('footprint', ['--footprint-ft', '4', '--footprint-taper-ft', '12'])):
        output = tmp_path / mode
        assert driver.main(['--reference', str(reference), '--output-dir', str(output), '--start', '1',
                            '--stop', '2', '--ensemble-size', '12', '--mda-steps', '3',
                            '--lateral-update', mode, *extra]) == 0
        with open(output / 'inversion_results_assim_1.pkl', 'rb') as stream:
            saved = pickle.load(stream)
        runs[mode] = (saved['prior_params'].reshape(2, nx, nz, -1), saved['post_mda_params'].reshape(2, nx, nz, -1))
        study = json.loads((output / 'study.json').read_text())
        assert study['lateral_update']['mode'] == mode
    prior, full = runs['full']
    for mode in ('column', 'footprint'):
        np.testing.assert_array_equal(runs[mode][0], prior)                       # same carried prior
        np.testing.assert_allclose(runs[mode][1][:, 2], full[:, 2])              # same bit-column trajectory
    _, column = runs['column']
    np.testing.assert_array_equal(column[:, [0, 4]], prior[:, [0, 4]])          # untouched beyond +-1 column
    assert not np.allclose(column[:, [1, 3]], prior[:, [1, 3]])
    _, footprint = runs['footprint']
    weights = driver.lateral_weights(nx, 5., 'footprint', footprint_ft=4., taper_ft=12.)
    assert 0 < weights[1] < 1 and 0 < weights[0] < weights[1]
    # The latent block is never clamped, so its increments are exactly the tapered bit-column increment
    # (ln Rh follows the same rule but is clamped to [-1, ln 40] after every step).
    bit_increment = footprint[1, 2] - prior[1, 2]
    for x in range(nx):
        np.testing.assert_allclose(footprint[1, x] - prior[1, x], weights[x] * bit_increment, atol=1e-10)
    assert np.all(footprint[0] <= np.log(40.) + 1e-12) and np.all(footprint[0] >= -1. - 1e-12)


def test_lateral_update_tiles_or_restricts_one_increment():
    nx, nz, members = 5, 3, 4
    rng = np.random.default_rng(8)
    previous = rng.normal(size=(2 * nx * nz, members))
    previous[:nx * nz] = rng.uniform(0., 2., size=(nx * nz, members))     # ln Rh inside [-1, ln 40]
    step = rng.normal(scale=.1, size=previous.shape)
    updated = previous + step
    weights = np.array([0., .5, 1., .5, 0.])
    tiled = driver.lateral_update(previous, updated.copy(), 'footprint', weights, nx, nz)
    increments = (tiled - previous).reshape(2, nx, nz, members)
    bit = step.reshape(2, nx, nz, members)[:, 2]
    for x in range(nx):
        np.testing.assert_allclose(increments[:, x], weights[x] * bit, atol=1e-12)
    restricted = driver.lateral_update(previous, updated.copy(), 'column', weights, nx, nz)
    np.testing.assert_allclose((restricted - previous).reshape(2, nx, nz, members),
                               weights[None, :, None, None] * step.reshape(2, nx, nz, members), atol=1e-12)
    np.testing.assert_array_equal(driver.lateral_update(previous, updated, 'full', weights, nx, nz), updated)
    capped = driver.lateral_update(previous, previous + 10., 'footprint', np.ones(nx), nx, nz)
    assert capped[:nx * nz].max() <= np.log(40.) + 1e-12


def test_fork_chain_can_keep_only_the_last_checkpoint(tmp_path, gate_environment):
    _, reference = gate_environment
    chain = run_chain(tmp_path, reference, 'source-last', '--on-gate', 'continue')
    (chain / 'laplace-problem-001-revert.npz').write_bytes(b'')
    (chain / 'laplace-problem-003-revert.npz').write_bytes(b'')
    fork = driver.fork_chain(chain, tmp_path / 'fork-last', 2, 'single point', first_index=2)
    assert (fork / 'laplace-problem-001-revert.npz').exists() and not (fork / 'laplace-problem-003-revert.npz').exists()
    assert sorted(p.name for p in fork.glob('inversion_results_assim_*.pkl')) == ['inversion_results_assim_2.pkl']
    assert [entry['logging_index'] for entry in json.loads((fork / 'history.json').read_text())] == [0, 1, 2]


def test_ridge_lateral_update_regresses_the_column_increment_onto_the_window(tmp_path, gate_environment):
    _, reference = gate_environment
    nx, nz = 5, 6
    saved = {}
    for mode in ('full', 'ridge'):
        output = tmp_path / f'ridge-{mode}'
        assert driver.main(['--reference', str(reference), '--output-dir', str(output), '--start', '1',
                            '--stop', '2', '--ensemble-size', '12', '--mda-steps', '3',
                            '--lateral-update', mode]) == 0
        with open(output / 'inversion_results_assim_1.pkl', 'rb') as stream:
            saved[mode] = pickle.load(stream)
    prior, post = saved['ridge']['prior_params'], saved['ridge']['post_mda_params']
    column, rest, propagate = driver.ridge_propagator(prior, nx, nz)
    np.testing.assert_array_equal(column, driver.bit_column_indices(nx, nz))
    np.testing.assert_allclose(post[column], saved['full']['post_mda_params'][column])   # same column trajectory
    expected = prior[rest] + propagate(post[column] - prior[column])
    latent = rest >= nx * nz                                   # never clamped
    np.testing.assert_allclose(post[rest][latent], expected[latent], atol=1e-8)
    assert not np.allclose(post[rest], saved['full']['post_mda_params'][rest])


def test_gate_export_uses_the_requested_2d_prior(tmp_path, gate_environment):
    _, reference = gate_environment
    chain = run_chain(tmp_path, reference, 'gate-ridge', '--lateral-update', 'ridge',
                      '--two-d-prior', 'ensemble_precision')
    problem = chain_laplace.load_problem(chain / 'laplace-problem-002.npz')
    assert problem['prior_type'] == 'ensemble_precision' and problem['likelihood_weight'] == 1.
    study = json.loads((chain / 'study.json').read_text())
    assert study['two_d_prior'] == 'ensemble_precision' and study['lateral_update']['mode'] == 'ridge'
    import run_chain_laplace_loop as loop
    arguments = loop.driver_arguments({**study, 'ensemble_size': 12}, 4)
    assert arguments[arguments.index('--two-d-prior') + 1] == 'ensemble_precision'
    assert arguments[arguments.index('--lateral-update') + 1] == 'ridge'
    assert arguments[arguments.index('--gate-threshold') + 1] == '10.0'
