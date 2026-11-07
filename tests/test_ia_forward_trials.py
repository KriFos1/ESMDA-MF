from threading import Barrier, Lock

import numpy as np
from types import SimpleNamespace

from esmda_remote2d import Remote2D, SimBackendConfig
import pilot_rml_index45 as runner


def test_four_dip_trials_are_parallel_forward_only_and_keep_failures(monkeypatch, tmp_path):
    barrier, lock, requests = Barrier(4, timeout=5), Lock(), []
    adapter = Remote2D(SimBackendConfig(concurrency=4, orientation='bedding'),
                       (2, 1, 2), 5., 5., 5., [('24kHz', '15.61ft')])

    class Backend:
        def trusts_ad(self, fidelity):
            return True
        def evaluate_batch(self, metadata, fields):
            with lock:
                requests.append(metadata.copy())
            barrier.wait()
            angle = metadata['apparent_dip_degrees']
            if angle > 10:
                raise RuntimeError('failed trial')
            return np.full((1, 8), angle), [np.empty((0, 0))]

    adapter._backend = Backend()
    well = {key: np.zeros((1, 1)) for key in ('X', 'Y', 'Z', 'Distance', 'Inclination', 'Azimuth')}
    physical = lambda raw: ({'rh': np.zeros(4), 'rv': np.zeros(4)}, None, None)
    monkeypatch.setattr(runner, 'evaluation_setup', lambda *args: (adapter, well, physical))
    study = {'dip_prior': {'bounds': [-45., 45.]}, 'index': 0, 'shift_ft': {'x': 0., 'y': 0., 'z': 0.}}
    inputs = {'mask': np.ones(8, dtype=bool), 'observations': np.zeros(8)}
    evaluate = runner.forward_evaluator(study, inputs, tmp_path)
    result = evaluate([np.array([x]) for x in (-.1, 0., .1, .3)])
    assert result.shape == (4, 8) and np.isnan(result[3]).all()
    assert all(not row['jacobian'] and not row['full_jacobian'] for row in requests)
    assert len({row['apparent_dip_degrees'] for row in requests}) == 4


def test_member_saves_accepted_jacobian_without_an_extra_solve(monkeypatch, tmp_path):
    (tmp_path / 'study.json').write_text('{}')
    study = {'count': 1, 'optimizer_method': runner.LM_OPTIMIZER, 'optimizer': {},
             'forward_only_candidates': True, 'save_jacobian': True}
    inputs = {'prior_centers': np.zeros((1, 3)), 'targets': np.zeros((1, 2)),
              'variance': np.ones(2)}
    monkeypatch.setattr(runner, 'read_study', lambda directory: (study, inputs))
    monkeypatch.setattr(runner, 'load_prior', lambda *args: object())
    calls = []
    def evaluate(model):
        calls.append('extra evaluation')
        raise AssertionError('The accepted Jacobian must be reused')
    evaluate.batch = lambda models: None
    monkeypatch.setattr(runner, 'model_evaluator', lambda *args: evaluate)
    forward = lambda models: None
    monkeypatch.setattr(runner, 'forward_evaluator', lambda *args: forward)
    jacobian = np.arange(6.).reshape(2, 3)
    def optimize(*args, **kwargs):
        assert kwargs['evaluate_forward'] is forward
        return {'model': np.ones(3), 'z': np.ones(3), 'prediction': np.ones(2), 'jacobian': jacobian,
                'reason': 'iteration_limit', 'converged': False, 'accepted_iterations': 1,
                'evaluations_this_attempt': 6, 'seconds_this_attempt': 1.}
    monkeypatch.setattr(runner, 'optimize_member_lm', optimize)
    runner.member(SimpleNamespace(run_dir=tmp_path, member=0))
    with np.load(tmp_path / 'member-000/laplace.npz') as saved:
        np.testing.assert_array_equal(saved['jacobian'], jacobian)
        np.testing.assert_array_equal(saved['prediction'], np.ones(2))
    assert not calls


def test_member_wires_the_tolerance_schedule_to_both_evaluators(monkeypatch, tmp_path):
    (tmp_path / 'study.json').write_text('{}')
    schedule = [[20., .01], [0., .001]]
    study = {'count': 1, 'optimizer_method': runner.LM_OPTIMIZER,
             'optimizer': {'max_iterations': 3, 'tolerance_schedule': schedule}, 'forward_only_candidates': True}
    inputs = {'prior_centers': np.zeros((1, 3)), 'targets': np.zeros((1, 2)), 'variance': np.ones(2)}
    monkeypatch.setattr(runner, 'read_study', lambda directory: (study, inputs))
    monkeypatch.setattr(runner, 'load_prior', lambda *args: object())
    seen = []
    evaluate = lambda model: None
    evaluate.batch = lambda models: None
    evaluate.set_tolerance = lambda value: seen.append(('jacobian', value))
    forward = lambda models: None
    forward.set_tolerance = lambda value: seen.append(('forward', value))
    monkeypatch.setattr(runner, 'model_evaluator', lambda *args: evaluate)
    monkeypatch.setattr(runner, 'forward_evaluator', lambda *args: forward)
    def optimize(*args, **kwargs):
        assert kwargs['tolerance_schedule'] == schedule and kwargs['max_iterations'] == 3
        kwargs['set_tolerance'](.01)
        return {'model': np.ones(3), 'z': np.ones(3), 'prediction': np.ones(2), 'reason': 'iteration_limit',
                'converged': False, 'accepted_iterations': 1, 'evaluations_this_attempt': 6,
                'seconds_this_attempt': 1., 'solver_tolerance': .01}
    monkeypatch.setattr(runner, 'optimize_member_lm', optimize)
    runner.member(SimpleNamespace(run_dir=tmp_path, member=0))
    assert seen == [('jacobian', .01), ('forward', .01)]
