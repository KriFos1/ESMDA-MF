import json

import numpy as np
import pytest

import lm_laplace
from esmda_remote2d import classify_solver_start_failure, is_solver_exit_174
from lm_laplace import SpatialPrior, optimize, sample_laplace, final_stage
from run_DA_sequential import engagement_invariant_windows


def test_structured_prior_apply_solve_and_sampling():
    rng = np.random.default_rng(19)
    ensemble = rng.normal(size=(12, 100))
    prior = SpatialPrior(ensemble, (3, 2), (5., 5.), (80., 75.))
    covariance = prior.apply(np.eye(12))
    np.testing.assert_allclose(prior.solve(covariance), np.eye(12), atol=1e-10)
    draws = prior.draw(30000, rng)
    np.testing.assert_allclose(np.cov(draws), covariance, atol=.035, rtol=.08)


def test_linear_gaussian_map_and_laplace_match_analytic_posterior():
    rng = np.random.default_rng(23)
    ensemble = rng.normal(scale=.3, size=(8, 400))
    prior = SpatialPrior(ensemble, (2, 2), (5., 5.), (20., 15.))
    jacobian = rng.normal(size=(3, 8))
    datum = np.array([.2, -.1, .3])
    variance = np.array([.1, .2, .3])
    covariance = prior.apply(np.eye(8))
    gain = covariance @ jacobian.T @ np.linalg.inv(jacobian @ covariance @ jacobian.T
                                                  + np.diag(variance))
    expected_mean = prior.mean + gain @ (datum - jacobian @ prior.mean)
    expected_covariance = covariance - gain @ jacobian @ covariance
    fit = optimize(prior.mean.copy(), prior, datum, variance,
                   lambda m: jacobian @ m, lambda m: jacobian,
                   max_step=1., max_iterations=30, relative_data_improvement=1e-6)
    np.testing.assert_allclose(fit['model'], expected_mean, atol=1e-5)
    assert fit['converged']
    assert np.all(np.diff(fit['objective_history']) < 0)
    draws = sample_laplace(fit, prior, variance, 30000, rng)
    np.testing.assert_allclose(draws.mean(axis=1), expected_mean, atol=.01)
    np.testing.assert_allclose(np.cov(draws), expected_covariance, atol=.006, rtol=.1)


def test_gate_keeps_1d_even_if_fitted_misfit_exceeds_threshold():
    rng = np.random.default_rng(29)
    ensemble = rng.normal(scale=.1, size=(8, 40))
    datum, variance = np.array([1.]), np.array([.01])
    zero_j = np.zeros((1, 8))
    two_d_j = np.zeros((1, 8)); two_d_j[0, 0] = 1.
    def forbidden_2d(*args):
        raise AssertionError('A 2D optimizer must never follow the 1D optimizer')

    # Pre-final misfit passes the gate; the fitted response still fails it.
    state, result = final_stage(
        ensemble, np.ones((1, 40)), datum, variance, 2., (2, 2), (5., 5.),
        (20., 15.), 4., lambda m: np.zeros(1), lambda m: zero_j,
        forbidden_2d, forbidden_2d, rng,
        max_iterations=30, max_step=1.)
    assert result['selected_fidelity'] == '1D'
    assert result['map_forward_misfit'] > 4.
    assert [x['fidelity'] for x in result['final_stage_history']] == ['1D']
    assert state.shape == ensemble.shape
    assert result['posterior_prediction_kind'] == 'forward'
    assert not np.allclose(state, ensemble)
    assert np.min(state.std(axis=1)) > 0


def test_gate_above_threshold_uses_only_2d_optimizer():
    rng = np.random.default_rng(31)
    ensemble = rng.normal(scale=.1, size=(8, 40))
    jacobian = np.zeros((1, 8))
    jacobian[0, 0] = 1.

    def forbidden_1d(*args):
        raise AssertionError('Pre-final gate chose 2D, so 1D LM must not run')

    state, result = final_stage(
        ensemble, np.zeros((1, 40)), np.ones(1), np.full(1, .01), 2.,
        (2, 2), (5., 5.), (20., 15.), 4., forbidden_1d, forbidden_1d,
        lambda m: jacobian @ m, lambda m: jacobian, rng,
        max_iterations=30, max_step=1., max_2d_starts=1)
    assert result['selected_fidelity'] == '2D'
    assert [x['fidelity'] for x in result['final_stage_history']] == ['2D']
    assert state.shape == ensemble.shape


def test_convergence_uses_less_than_five_percent_actual_data_improvement():
    prior = SpatialPrior(np.zeros((8, 40)), (2, 2), (5., 5.), (20., 15.))
    jac = np.zeros((1, 8)); jac[0, 0] = 1.
    fit = optimize(np.zeros(8), prior, np.ones(1), np.full(1, 2.),
                   lambda m: jac @ m, lambda m: jac)
    assert fit['converged']
    assert fit['reason'] == 'relative_data_improvement'
    assert len(fit['objective_history']) == 2
    old, new = fit['data_misfit_history']
    assert 0 < (old - new) / old < .05
    assert fit['relative_data_improvement_history'][-1] == (old - new) / old


def test_map_optimizer_respects_optional_rh_cap():
    prior = SpatialPrior(np.zeros((8, 40)), (2, 2), (5., 5.), (20., 15.))
    jac = np.zeros((1, 8)); jac[0, 0] = 1.
    visited = []

    def forward(model):
        visited.append(model.copy())
        return jac @ model

    optimize(np.zeros(8), prior, np.array([10.]), np.array([.01]), forward,
             lambda model: jac, max_iterations=1, max_trials=1, max_step=10.,
             rh_limits=(-1., np.log(2.)))
    assert len(visited) >= 2
    assert max(model[0] for model in visited) <= np.log(2.) + 1e-12


def test_map_optimizer_can_use_unclipped_latent_gaussian_coordinates():
    prior = SpatialPrior(np.zeros((8, 40)), (2, 2), (5., 5.), (20., 15.))
    jac = np.zeros((1, 8)); jac[0, 0] = 1.
    visited = []

    def forward(model):
        visited.append(model[0])
        return jac @ model

    optimize(np.zeros(8), prior, np.array([10.]), np.array([.01]), forward,
             lambda model: jac, max_iterations=1, max_trials=1, max_step=10.,
             rh_limits=None)
    assert max(visited) > np.log(2.)


def test_worsening_data_fit_is_not_convergence_despite_lower_map_objective():
    prior = SpatialPrior(np.zeros((8, 40)), (2, 2), (5., 5.), (20., 15.))
    jac = np.zeros((1, 8)); jac[0, 0] = 1.
    initial = np.zeros(8); initial[0] = .5
    fit = optimize(initial, prior, np.ones(1), np.ones(1),
                   lambda m: jac @ m, lambda m: jac, max_iterations=1)
    assert fit['objective_history'][-1] < fit['objective_history'][0]
    assert fit['data_misfit_history'][-1] > fit['data_misfit_history'][0]
    assert not fit['converged']


def failed_batch(tmp_path, exit_status=174, error=None):
    batch_id = 'a' * 32
    directory = tmp_path / batch_id
    directory.mkdir(exist_ok=True)
    status = {'batch_id': batch_id, 'state': 'FAILED', 'connected': True,
              'evaluations': [{'batch_id': batch_id, 'state': 'FAILED',
                               'error': (error if error is not None else
                                         f'CalledProcessError: returned non-zero exit status {exit_status}.')}]}
    (directory / 'status.json').write_text(json.dumps(status))
    return RuntimeError(f'Batch {batch_id}: FAILED. Details: {directory}/status.json')


def test_exit_174_requires_a_confirmed_single_field_failure(tmp_path):
    failure = failed_batch(tmp_path)
    assert is_solver_exit_174(failure)
    status_path = tmp_path / ('a' * 32) / 'status.json'
    status = json.loads(status_path.read_text())
    for change in ({'state': 'DISCONNECTED'}, {'connected': False},
                   {'batch_id': 'b' * 32}, {'evaluations': status['evaluations'] * 2}):
        status_path.write_text(json.dumps({**status, **change}))
        assert not is_solver_exit_174(failure)
    status_path.write_text('not valid json')
    assert not is_solver_exit_174(failure)
    assert not is_solver_exit_174(failed_batch(tmp_path, exit_status=1))
    assert not is_solver_exit_174(RuntimeError('Batch failed with exit status 174'))


def test_initial_solver_convergence_error_is_distinct_from_transport(tmp_path):
    failure = failed_batch(
        tmp_path, error='RuntimeError: Solver reported a convergence error; see convergence_error.txt')
    assert classify_solver_start_failure(failure) == 'solver_convergence'
    assert not is_solver_exit_174(failure)
    assert classify_solver_start_failure(failed_batch(tmp_path, exit_status=135)) == 'exit_135'
    assert classify_solver_start_failure(RuntimeError('Reconnect with pxwf-jobs status')) is None


def test_top_five_2d_starts_are_all_optimized_and_best_final_misfit_is_selected(monkeypatch):
    ensemble = np.zeros((8, 6))
    ensemble[0] = np.arange(6) / 10.
    predictions = np.array([[10., 9., 8., 7., 6., 5.]])
    optimized_starts = []
    forward_starts = []
    final_predictions = {5: 1., 4: .5, 3: .1, 2: .3, 1: .2}

    def fake_optimize_batch(initials, initial_predictions, *args, **options):
        results = []
        for initial in initials:
            member = int(round(initial[0] * 10))
            optimized_starts.append(member)
            results.append({'model': initial.copy(),
                            'prediction': np.array([final_predictions[member]]),
                            'jacobian': np.zeros((1, len(initial))),
                            'objective_history': [1., .5], 'converged': True,
                            'reason': 'relative_data_improvement',
                            'data_misfit_history': [1., .01],
                            'relative_data_improvement_history': [.01],
                            'trial_errors': [], 'seconds': .1})
        return results

    monkeypatch.setattr(lm_laplace, 'optimize_batch', fake_optimize_batch)

    def forward_2d(model):
        forward_starts.append(int(round(model[0] * 10)))
        return np.zeros(1)

    state, result = final_stage(
        ensemble, predictions, np.zeros(1), np.ones(1), 1., (2, 2), (5., 5.),
        (20., 15.), 4., lambda m: np.zeros(1), lambda m: np.zeros((1, 8)),
        forward_2d, lambda m: np.zeros((1, 8)), np.random.default_rng(35),
        max_2d_starts=5)
    assert optimized_starts == forward_starts == [5, 4, 3, 2, 1]
    assert result['gate_selected_fidelity'] == result['selected_fidelity'] == '2D'
    assert result['two_d_candidate_count'] == result['two_d_converged_candidate_count'] == 5
    assert result['required_converged_2d_starts'] == 1
    assert result['two_d_start_count'] == 5
    assert result['best_member'] == 5
    assert result['map_start_member'] == 3
    assert result['map_forward_misfit'] == pytest.approx(.01)
    assert result['map_converged'] and result['laplace_at_converged_map']
    assert [x['fidelity'] for x in result['final_stage_history']] == ['2D'] * 5
    assert state.shape == ensemble.shape


def test_one_converged_start_is_enough_and_unconverged_lower_misfit_is_ignored(monkeypatch):
    ensemble = np.zeros((8, 6))
    ensemble[0] = np.arange(6) / 10.
    predictions = np.array([[10., 9., 8., 7., 6., 5.]])
    optimize_calls = []
    one_d_forwards = []

    def fake_optimize_batch(initials, initial_predictions, *args, **options):
        results = []
        for initial in initials:
            member = int(round(initial[0] * 10))
            optimize_calls.append((member, True))
            map_prediction = {5: .5, 4: .1, 3: .3, 2: 0., 1: .4}[member]
            results.append({'model': initial.copy(), 'prediction': np.array([map_prediction]),
                            'jacobian': np.zeros((1, len(initial))), 'objective_history': [1.],
                            'converged': member != 2,
                            'reason': 'iteration_limit' if member == 2 else 'relative_data_improvement',
                            'data_misfit_history': [map_prediction ** 2],
                            'relative_data_improvement_history': [],
                            'trial_errors': [], 'seconds': .1})
        return results

    def fake_optimize_1d(initial, prior, datum, variance, forward, derivative, **options):
        optimize_calls.append((int(round(initial[0] * 10)), False))
        raise AssertionError('At least one 2D start converged; 1D fallback is not allowed')

    monkeypatch.setattr(lm_laplace, 'optimize_batch', fake_optimize_batch)
    monkeypatch.setattr(lm_laplace, 'optimize', fake_optimize_1d)

    def forward_1d(model):
        one_d_forwards.append(True)
        return np.zeros(1)

    _, result = final_stage(
        ensemble, predictions, np.zeros(1), np.ones(1), 1., (2, 2), (5., 5.),
        (20., 15.), 4., forward_1d, lambda m: np.zeros((1, 8)),
        lambda m: np.zeros(1), lambda m: np.zeros((1, 8)), np.random.default_rng(36),
        max_2d_starts=5)
    assert optimize_calls == [(5, True), (4, True), (3, True), (2, True), (1, True)]
    assert result['gate_selected_fidelity'] == '2D'
    assert result['selected_fidelity'] == '2D'
    assert result['two_d_candidate_count'] == 5
    assert result['two_d_converged_candidate_count'] == 4
    assert result['required_converged_2d_starts'] == 1
    assert result['map_start_member'] == 4
    assert result['map_forward_misfit'] == pytest.approx(.01)
    assert result['fallback_reason'] is None
    assert result['posterior_prediction_kind'] == 'linearized'
    assert not result['posterior_forward_verified']
    assert not one_d_forwards


def test_1d_fallback_occurs_only_when_no_2d_start_converges(monkeypatch):
    ensemble = np.zeros((8, 6))
    ensemble[0] = np.arange(6) / 10.
    predictions = np.array([[10., 9., 8., 7., 6., 5.]])
    calls = []

    def fake_optimize_batch(initials, initial_predictions, *args, **options):
        calls.extend(['2D'] * len(initials))
        return [{'model': initial.copy(), 'prediction': np.zeros(1),
                 'jacobian': np.zeros((1, len(initial))), 'objective_history': [1.],
                 'converged': False, 'reason': 'iteration_limit',
                 'data_misfit_history': [0.], 'relative_data_improvement_history': [],
                 'trial_errors': [], 'seconds': .1} for initial in initials]

    def fake_optimize_1d(initial, prior, datum, variance, forward, derivative, **options):
        calls.append('1D')
        return {'model': initial.copy(), 'prediction': np.zeros(1),
                'jacobian': np.zeros((1, len(initial))), 'objective_history': [1.],
                'converged': True, 'reason': 'relative_data_improvement',
                'data_misfit_history': [0.], 'relative_data_improvement_history': [],
                'trial_errors': [], 'seconds': .1}

    monkeypatch.setattr(lm_laplace, 'optimize_batch', fake_optimize_batch)
    monkeypatch.setattr(lm_laplace, 'optimize', fake_optimize_1d)
    _, result = final_stage(
        ensemble, predictions, np.zeros(1), np.ones(1), 1., (2, 2), (5., 5.),
        (20., 15.), 4., lambda m: np.zeros(1), lambda m: np.zeros((1, 8)),
        lambda m: np.zeros(1), lambda m: np.zeros((1, 8)), np.random.default_rng(37),
        max_2d_starts=5)
    assert calls == ['2D'] * 5 + ['1D']
    assert result['two_d_candidate_count'] == 5
    assert result['two_d_converged_candidate_count'] == 0
    assert result['selected_fidelity'] == '1D'
    assert result['fallback_reason'] == 'no_converged_2d_start'
    assert result['map_start_kind'] == '1d_fallback'


def test_batch_2d_orchestration_batches_active_candidates_and_skips_failed_start():
    ensemble = np.zeros((8, 6))
    ensemble[0] = np.arange(6) / 10.
    predictions = np.array([[10., 9., 8., 7., 6., 5.]])
    forward_batch_sizes, jacobian_batch_sizes = [], []

    def forward_batch(models):
        forward_batch_sizes.append(len(models))
        responses = [np.zeros(1) for _ in models]
        if len(forward_batch_sizes) == 1:
            responses[0] = None
        return responses

    def jacobian_batch(models):
        jacobian_batch_sizes.append(len(models))
        return [np.zeros((1, 8)) for _ in models]

    _, result = final_stage(
        ensemble, predictions, np.zeros(1), np.ones(1), 1., (2, 2), (5., 5.),
        (20., 15.), 4., lambda m: pytest.fail('2D gate must not call 1D'),
        lambda m: pytest.fail('2D gate must not call 1D Jacobian'),
        lambda m: pytest.fail('batch path must not call scalar 2D forward'),
        lambda m: pytest.fail('batch path must not call scalar 2D Jacobian'),
        np.random.default_rng(38), max_2d_starts=5,
        forward_2d_batch=forward_batch, jacobian_2d_batch=jacobian_batch)
    assert forward_batch_sizes == [5, 4]
    assert jacobian_batch_sizes == [4, 4]
    assert result['selected_fidelity'] == '2D'
    assert result['map_start_member'] == 4
    assert result['two_d_candidate_count'] == 4
    assert result['two_d_converged_candidate_count'] == 4
    assert result['two_d_initial_failures'][0]['member'] == 5


def test_five_2d_startup_failures_fall_back_to_1d_and_regenerate_ensemble(tmp_path):
    ensemble = np.zeros((8, 6))
    ensemble[0] = np.arange(6) / 10.
    calls = []
    failure = failed_batch(tmp_path)

    def forward_2d(model):
        calls.append(int(round(model[0] * 10)))
        raise failure

    state, result = final_stage(
        ensemble, np.arange(6)[None] / 10., np.ones(1), np.full(1, .01), 2.,
        (2, 2), (5., 5.), (20., 15.), 4.,
        lambda m: np.zeros(1), lambda m: np.zeros((1, 8)),
        forward_2d, lambda m: np.zeros((1, 8)), np.random.default_rng(41),
        max_iterations=0, is_recoverable_2d_failure=is_solver_exit_174)
    assert calls == [5, 4, 3, 2, 1]
    assert result['best_member'] == result['map_start_member'] == 5
    assert result['gate_selected_fidelity'] == '2D'
    assert result['selected_fidelity'] == '1D'
    assert result['two_d_mode'] == 'abandoned'
    assert result['fallback_reason'] == '2d_starts_exit_174'
    assert len(result['two_d_initial_failures']) == 5
    assert [entry['fidelity'] for entry in result['final_stage_history']] == ['2D'] * 5 + ['1D']
    assert result['posterior_forward_verified']
    assert state.shape == ensemble.shape
    assert not result['map_converged']


def test_all_2d_start_failures_fall_back_to_1d_with_invariant_first(tmp_path):
    ensemble = np.zeros((8, 6)); ensemble[0] = np.arange(6) / 10.
    calls = []
    failure = failed_batch(tmp_path)

    def forward_2d(model):
        calls.append(float(model[0]))
        raise failure

    _, result = final_stage(
        ensemble, np.arange(6)[None] / 10., np.ones(1), np.full(1, .01), 2.,
        (2, 2), (5., 5.), (20., 15.), 4.,
        lambda m: np.zeros(1), lambda m: np.zeros((1, 8)),
        forward_2d, lambda m: np.zeros((1, 8)), np.random.default_rng(45),
        max_iterations=0, is_recoverable_2d_failure=classify_solver_start_failure,
        invariant_start=lambda m: engagement_invariant_windows(m[:, None], (2, 5, 2))[:, 0])
    assert calls == [0., .5, .4, .3, .2, .1]
    assert result['selected_fidelity'] == '1D'
    assert result['gate_selected_fidelity'] == '2D'
    assert result['invariant_start_attempted']
    assert [item['start_kind'] for item in result['two_d_initial_failures']] == [
        'invariant_extension', *['ranked_member'] * 5]
    assert result['two_d_mode'] == 'abandoned'


def test_recoverable_initial_jacobian_failure_does_not_abort_other_starts(tmp_path):
    ensemble = np.zeros((8, 6))
    ensemble[0] = np.arange(6) / 10.
    predictions = np.array([[10., 9., 8., 7., 6., 5.]])
    failure = failed_batch(tmp_path)
    calls = []

    def forward_2d(model):
        calls.append(('forward', int(round(model[0] * 10))))
        return np.zeros(1)

    def jacobian_2d(model):
        member = int(round(model[0] * 10))
        if member == 5:
            raise failure
        return np.zeros((1, 8))

    _, result = final_stage(
        ensemble, predictions, np.zeros(1), np.ones(1), 1., (2, 2), (5., 5.),
        (20., 15.), 4., lambda m: np.zeros(1), lambda m: np.zeros((1, 8)),
        forward_2d, jacobian_2d, np.random.default_rng(43), max_iterations=1,
        is_recoverable_2d_failure=classify_solver_start_failure, max_2d_starts=5)
    assert sorted({member for kind, member in calls if kind == 'forward'}, reverse=True) == [5, 4, 3, 2, 1]
    assert sum(kind == 'forward' for kind, _ in calls) == 9
    assert result['two_d_candidate_count'] == 4
    assert result['two_d_converged_candidate_count'] == 4
    assert result['two_d_initial_failures'][0]['member'] == 5
    assert result['two_d_initial_failures'][0]['stage'] == 'initial_jacobian'
    assert result['selected_fidelity'] == '2D'
    assert result['map_start_member'] == 4
    assert result['fallback_reason'] is None


def test_non_174_initial_2d_failure_propagates(tmp_path):
    failure = failed_batch(tmp_path, exit_status=1)

    def forward_2d(model):
        raise failure

    with pytest.raises(RuntimeError, match='Batch .* FAILED'):
        final_stage(np.zeros((8, 6)), np.zeros((1, 6)), np.ones(1), np.ones(1), 2.,
                    (2, 2), (5., 5.), (20., 15.), .5,
                    lambda m: np.zeros(1), lambda m: np.zeros((1, 8)),
                    forward_2d, lambda m: np.zeros((1, 8)), np.random.default_rng(47),
                    max_iterations=0, is_recoverable_2d_failure=is_solver_exit_174)
