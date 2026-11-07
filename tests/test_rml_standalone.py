import json

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from background_prior import BackgroundPrior
from rml_standalone import (factor_apply, gauss_newton_step, objective_gradient, optimize_member,
                            randomized_inputs)


def prior_fixture():
    return BackgroundPrior((2, 2), (5., 5.), ((4., 6.), (3., 7.)),
                           (.2, 0.), (.5, .2))


def test_factor_matches_covariance_and_transpose():
    prior = prior_fixture()
    factor = factor_apply(prior, np.eye(8))
    np.testing.assert_allclose(factor @ factor.T, prior.apply(np.eye(8)), atol=1e-15)
    np.testing.assert_allclose(factor_apply(prior, np.eye(8), transpose=True), factor.T)
    offset = factor @ np.arange(8.)
    assert prior.quadratic(offset) == pytest.approx(np.arange(8.) @ np.arange(8.))


def test_randomizations_are_independent_and_member_stable():
    prior = prior_fixture()
    observations, variance = np.array([.3, -.2]), np.array([.01, .04])
    first = randomized_inputs(prior, observations, variance, 3, 2045)
    all_members = randomized_inputs(prior, observations, variance, 25, 2045)
    for name in ('prior_centers', 'measurement_noise', 'targets'):
        np.testing.assert_array_equal(first[name], all_members[name][:3])
    np.testing.assert_array_equal(first['targets'], observations + first['measurement_noise'])
    assert len(np.unique(all_members['prior_centers'], axis=0)) == 25


def test_full_nonlinear_posterior_gradient():
    prior = prior_fixture()
    rng = np.random.default_rng(51)
    center, z = rng.normal(size=(2, 8))
    matrix = rng.normal(size=(3, 8))
    target, variance = np.array([.1, .4, -.3]), np.array([.1, .2, .3])
    def evaluate(model):
        values = matrix @ model
        return np.sin(values), np.cos(values)[:, None] * matrix
    def objective(point):
        return objective_gradient(point, center, target, variance, prior, evaluate)
    expected = approx_derivative(lambda point: objective(point)[0], z).ravel()
    np.testing.assert_allclose(objective(z)[1], expected, atol=1e-8, rtol=1e-7)


def test_25_fits_match_exact_linear_gaussian_randomized_minimizers(tmp_path):
    prior = prior_fixture()
    rng = np.random.default_rng(28)
    matrix = rng.normal(size=(3, 8))
    observations, variance = np.array([.1, -.3, .2]), np.array([.02, .03, .04])
    inputs = randomized_inputs(prior, observations, variance, 25, 2045)
    covariance = prior.apply(np.eye(8))
    gain = covariance @ matrix.T @ np.linalg.inv(
        matrix @ covariance @ matrix.T + np.diag(variance))
    for member, (center, target) in enumerate(zip(inputs['prior_centers'], inputs['targets'])):
        fit = optimize_member(prior, center, target, variance,
                              lambda model: (matrix @ model, matrix),
                              checkpoint=tmp_path / f'{member}.npz', identity='linear',
                              max_iterations=150, max_trials=30, gradient_tolerance=1e-6)
        expected = center + gain @ (target - matrix @ center)
        np.testing.assert_allclose(fit['model'], expected, atol=2e-6)
        assert fit['converged'], fit['message']
        assert fit['accepted_iterations'] == 1
        assert fit['jacobian_evaluations_this_attempt'] == 2
        assert fit['history'][1]['gain_ratio'] == pytest.approx(1.)
        assert fit['history'][0]['prior_loss'] == 0.
        assert np.all(np.diff([row['objective'] for row in fit['history']]) <= 0)


def test_resume_preserves_randomized_problem_and_iteration_budget(tmp_path):
    prior = prior_fixture()
    matrix = np.random.default_rng(4).normal(size=(3, 8))
    center, target, variance = np.ones(8), np.ones(3), np.full(3, .01)
    path = tmp_path / 'checkpoint.npz'
    def evaluate(model):
        values = matrix @ model
        return np.exp(values / 5.), np.exp(values / 5.)[:, None] * matrix / 5.
    partial = optimize_member(prior, center, target, variance, evaluate,
                              checkpoint=path, identity='same', max_iterations=1)
    assert not partial['converged']
    resumed = optimize_member(prior, center, target, variance, evaluate,
                              checkpoint=path, identity='same', max_iterations=100,
                              max_trials=30, gradient_tolerance=1e-6)
    assert resumed['converged']
    assert resumed['history'][:2] == partial['history']
    with pytest.raises(ValueError, match='different inputs'):
        optimize_member(prior, center, target, variance, evaluate,
                        checkpoint=path, identity='changed')
    with np.load(path, allow_pickle=False) as checkpoint:
        assert json.loads(str(checkpoint['history'])) == resumed['history']


@pytest.mark.parametrize('damping', [0., .1, 10.])
def test_data_space_step_solves_full_gauss_newton_normal_equations(damping):
    rng = np.random.default_rng(93)
    sensitivity = rng.normal(size=(3, 17))
    z, residual = rng.normal(size=17), rng.normal(size=3)
    gradient = z + sensitivity.T @ residual
    expected = np.linalg.solve((1 + damping) * np.eye(17) + sensitivity.T @ sensitivity,
                               -gradient)
    np.testing.assert_allclose(gauss_newton_step(z, residual, sensitivity, damping),
                               expected, atol=1e-14)


def test_damping_reuses_current_linearization_and_checkpointed_jacobian(tmp_path):
    prior = prior_fixture()
    center, target, variance = np.zeros(8), np.array([2.]), np.array([.01])
    calls = []
    def evaluate(model):
        calls.append(model.copy())
        prediction = np.array([np.exp(5 * model[0])])
        jacobian = np.zeros((1, 8))
        jacobian[0, 0] = 5 * prediction[0]
        return prediction, jacobian
    path = tmp_path / 'gn.npz'
    partial = optimize_member(prior, center, target, variance, evaluate,
                              checkpoint=path, identity='GN', max_iterations=1)
    initial_calls = len(calls)
    assert partial['accepted_iterations'] == 1
    resumed = optimize_member(prior, center, target, variance, evaluate,
                              checkpoint=path, identity='GN', max_iterations=100)
    assert resumed['converged']
    # Every evaluation is a new candidate: the accepted Jacobian is never recomputed.
    assert len(calls) == len(partial['trial_history']) + 1 + resumed['evaluations_this_attempt']
    assert resumed['evaluations_this_attempt'] == len(resumed['trial_history']) - len(partial['trial_history'])
    assert len(calls) > initial_calls
    for left, right in zip(calls, calls[1:]):
        assert not np.array_equal(left, right)


def test_nonlinear_rejected_steps_raise_damping_without_changing_accepted_state(tmp_path):
    prior = prior_fixture()
    center, target, variance = np.zeros(8), np.array([10.]), np.array([.01])
    def evaluate(model):
        prediction = np.array([np.exp(5 * model[0])])
        jacobian = np.zeros((1, 8))
        jacobian[0, 0] = 5 * prediction[0]
        return prediction, jacobian
    partial = optimize_member(prior, center, target, variance, evaluate,
                              checkpoint=tmp_path / 'rejected.npz', max_trials=1)
    assert partial['reason'] == 'no_acceptable_trial'
    np.testing.assert_array_equal(partial['z'], np.zeros(8))
    assert partial['next_damping'] > 0
    assert not partial['trial_history'][0]['accepted']
    assert partial['history'][0]['objective'] == pytest.approx(.5 * 9 ** 2 / .01)
    resumed = optimize_member(prior, center, target, variance, evaluate,
                              checkpoint=tmp_path / 'rejected.npz', max_trials=20,
                              max_iterations=100)
    assert resumed['converged']
    assert resumed['history'][0] == partial['history'][0]
    assert resumed['history'][1]['damping'] > partial['next_damping']
    assert np.all(np.diff([row['objective'] for row in resumed['history']]) < 0)


def test_time_budget_returns_last_accepted_prediction(monkeypatch, tmp_path):
    import rml_standalone
    clock = iter([0., 0., 2., 2.])
    monkeypatch.setattr(rml_standalone.time, 'monotonic', lambda: next(clock))
    matrix = np.ones((1, 8))
    fit = optimize_member(prior_fixture(), np.zeros(8), np.ones(1), np.ones(1),
                          lambda model: (matrix @ model, matrix),
                          checkpoint=tmp_path / 'deadline.npz', max_seconds=1.)
    assert fit['reason'] == 'time_limit'
    assert fit['jacobian_evaluations_this_attempt'] == 1
    np.testing.assert_array_equal(fit['prediction'], np.zeros(1))
    np.testing.assert_array_equal(fit['z'], np.zeros(8))


def test_checkpoint_rejects_legacy_bfgs_state(tmp_path):
    path = tmp_path / 'old.npz'
    np.savez(path, identity='same', z=np.zeros(8), history='[]')
    with pytest.raises(ValueError, match='different optimizer'):
        optimize_member(prior_fixture(), np.zeros(8), np.zeros(1), np.ones(1),
                        lambda model: None, checkpoint=path, identity='same')


def test_physical_cap_and_anisotropy_chain_rule(monkeypatch, tmp_path):
    import esmda_remote2d
    import pilot_rml_index45 as runner

    weights = np.arange(1., 9.).reshape(2, 4)
    # One tool: rows of ``weights`` on UADP (phase), UADA and UHRA (attenuations).
    response = np.zeros((8, 4))
    response[[2, 3, 5]] = weights[[0, 1, 0]]
    class Backend:
        def trusts_ad(self, fidelity):
            return True
    class Adapter:
        def backend(self):
            return Backend()
        def run_grad_multi(self, parameter_list, index, shift, well):
            results = []
            for parameters in parameter_list:
                # Single y-layer: parameter dictionaries are x-fastest Fortran order.
                rh = parameters['rh'].reshape(2, 1, 2, order='F')[:, 0, :].ravel()
                rv = parameters['rv'].reshape(2, 1, 2, order='F')[:, 0, :].ravel()
                results.append((response @ (rh + rv), response, response))
            return results
    monkeypatch.setattr(esmda_remote2d, 'make_remote2d', lambda *args: Adapter())
    import run_DA_sequential as sequential
    # model_evaluator configures module globals; restore them after the test.
    for name in ('dims', 'dX', 'dY', 'dZ', 'min_ratio', 'max_ratio', 'log_rv_cap', 'anisotropy'):
        monkeypatch.setattr(sequential, name, getattr(sequential, name))
    monkeypatch.setenv('SLURM_JOB_ID', 'test')
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '5')
    study = {'dims': [2, 1, 2], 'spacing_ft': [5., 5., 5.], 'ratio_min': 1.,
             'ratio_max': 3., 'log_rv_cap': np.log(40.), 'rh_limits': [-1., np.log(40.)],
             'simulator': {'backend': 'local', 'cpus': 5, 'concurrency': 1},
             'tools': [], 'index': 45, 'shift_ft': {}}
    amplitudes = np.isin(np.arange(8), [3, 5])
    evaluate = runner.model_evaluator(study, {'mask': amplitudes, 'observations': np.zeros(2)}, tmp_path)
    raw = np.array([-2., 0., 3.4, 5., -.1, .1, 0., .2])
    prediction, actual = evaluate(raw)
    expected = approx_derivative(lambda model: evaluate(model)[0], raw)
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-7)
    assert np.all(actual[:, [0, 3]] == 0)
    batch = evaluate.batch([raw, raw + .01])
    np.testing.assert_array_equal(batch[0][1], actual)
    assert len(batch) == 2
    # A phase channel is moved to the observed branch; its Jacobian row is untouched.
    with_phase = np.isin(np.arange(8), [2, 3])
    observations = np.array([prediction[1] + 720. + 30., 0.])
    shifted, jacobian = runner.model_evaluator(study, {'mask': with_phase, 'observations': observations},
                                               tmp_path)(raw)
    np.testing.assert_allclose(shifted, [prediction[1] + 720., prediction[0]])
    np.testing.assert_array_equal(jacobian, actual[[1, 0]])


from rml_standalone import (LM_OPTIMIZER, candidate_dampings, optimize_member_lm, step_size,
                            trust_damping)


def batch_of(evaluate, calls=None):
    def evaluate_batch(models):
        if calls is not None:
            calls.append(len(models))
        return [evaluate(model) for model in models]
    return evaluate_batch


def test_trust_damping_meets_radius_and_candidates_are_nested():
    prior = prior_fixture()
    rng = np.random.default_rng(12)
    sensitivity, z, residual = rng.normal(size=(3, 8)), np.zeros(8), rng.normal(size=3) * 20
    gram = sensitivity @ sensitivity.T
    assert step_size(prior, gauss_newton_step(z, residual, sensitivity, 1e-6, gram)) > .3
    damping = trust_damping(prior, z, residual, sensitivity, gram, .3, 1e-6)
    assert step_size(prior, gauss_newton_step(z, residual, sensitivity, damping, gram)) <= .3
    assert step_size(prior, gauss_newton_step(z, residual, sensitivity, damping / 1.02, gram)) > .3
    dampings = candidate_dampings(prior, z, residual, sensitivity, gram, .3, 1e-6, 4)
    assert np.all(np.diff(dampings) > 0)
    sizes = [step_size(prior, gauss_newton_step(z, residual, sensitivity, value, gram)) for value in dampings]
    assert all(size <= .3 / 2 ** index + 1e-12 for index, size in enumerate(sizes))
    # A step already inside the radius keeps the floor damping.
    assert trust_damping(prior, z, residual * 1e-6, sensitivity, gram, .3, 1e-6) == 1e-6


def test_lm_reaches_exact_linear_gaussian_minimizers_with_four_parallel_candidates(tmp_path):
    prior = prior_fixture()
    rng = np.random.default_rng(28)
    matrix = rng.normal(size=(3, 8))
    observations, variance = np.array([.1, -.3, .2]), np.array([.02, .03, .04])
    inputs = randomized_inputs(prior, observations, variance, 5, 2045)
    covariance = prior.apply(np.eye(8))
    gain = covariance @ matrix.T @ np.linalg.inv(matrix @ covariance @ matrix.T + np.diag(variance))
    for member, (center, target) in enumerate(zip(inputs['prior_centers'], inputs['targets'])):
        calls = []
        fit = optimize_member_lm(prior, center, target, variance,
                                 batch_of(lambda model: (matrix @ model, matrix), calls),
                                 checkpoint=tmp_path / f'{member}.npz', identity='linear',
                                 relative_gradient_tolerance=1e-9, stagnation_tolerance=1e-12,
                                 max_iterations=200)
        expected = center + gain @ (target - matrix @ center)
        np.testing.assert_allclose(fit['model'], expected, atol=1e-6)
        assert fit['converged'] and fit['optimizer'] == LM_OPTIMIZER
        assert calls[0] == 1 and set(calls[1:]) == {4}
        objectives = [row['objective'] for row in fit['history']]
        assert np.all(np.diff(objectives) < 0)
        assert all(row['step_inf'] <= trust + 1e-12 for row, trust in
                   zip(fit['history'][1:], [.3] + [row['trust'] for row in fit['history'][1:]]))


def test_lm_first_step_respects_initial_trust_radius(tmp_path):
    prior = prior_fixture()
    center, target, variance = np.zeros(8), np.array([5.]), np.array([.01])
    matrix = np.ones((1, 8))
    fit = optimize_member_lm(prior, center, target, variance,
                             batch_of(lambda model: (matrix @ model, matrix)),
                             max_iterations=1, initial_trust=.3)
    assert fit['history'][1]['step_inf'] <= .3 + 1e-12
    first_round = [row for row in fit['trial_history'] if row['iteration'] == 1 and row['round'] == 1]
    assert len(first_round) == 4
    assert all(row['step_inf'] <= .3 / 2 ** index + 1e-12 for index, row in enumerate(first_round))


def test_lm_stops_on_objective_stagnation_and_failed_candidates_are_rejected(tmp_path):
    prior = prior_fixture()
    center, target, variance = np.zeros(8), np.array([1.]), np.array([.01])
    counter = {'n': 0}
    def evaluate(model):
        counter['n'] += 1
        if counter['n'] % 4 == 2:
            return None
        prediction = np.array([np.tanh(model[0])])
        jacobian = np.zeros((1, 8))
        jacobian[0, 0] = 1 - prediction[0] ** 2
        return prediction, jacobian
    def evaluate_batch(models):
        return [evaluate(model) for model in models]
    fit = optimize_member_lm(prior, center, target, variance, evaluate_batch, max_iterations=100)
    assert fit['reason'] in ('objective_stagnation', 'relative_gradient')
    assert any(row['failed'] and not row['accepted_candidate'] for row in fit['trial_history'])
    assert np.all(np.diff([row['objective'] for row in fit['history']]) < 0)


def test_lm_reports_nonlinearity_limited_when_no_candidate_decreases(tmp_path):
    prior = prior_fixture()
    center, target, variance = np.zeros(8), np.array([1.]), np.array([.01])
    jacobian = np.ones((1, 8))
    def evaluate(model):
        # The linear model predicts descent; the actual objective always rises.
        return np.array([0. if not np.any(model) else -1.]), jacobian
    fit = optimize_member_lm(prior, center, target, variance, batch_of(evaluate),
                             checkpoint=tmp_path / 'nl.npz', max_rounds=3)
    assert fit['reason'] == 'nonlinearity_limited' and not fit['converged']
    assert fit['accepted_iterations'] == 0
    rounds = {row['round'] for row in fit['trial_history']}
    assert rounds == {1, 2, 3}
    trusts = [row['trust'] for row in fit['trial_history'][::4]]
    assert trusts[0] == .3 and trusts[1] < trusts[0] and trusts[2] < trusts[1]


def test_lm_checkpoint_restart_and_identity(tmp_path):
    prior = prior_fixture()
    matrix = np.random.default_rng(4).normal(size=(3, 8))
    center, target, variance = np.ones(8), np.ones(3), np.full(3, .01)
    def evaluate(model):
        values = matrix @ model
        return np.exp(values / 5.), np.exp(values / 5.)[:, None] * matrix / 5.
    path = tmp_path / 'lm.npz'
    partial = optimize_member_lm(prior, center, target, variance, batch_of(evaluate),
                                 checkpoint=path, identity='same', max_iterations=1)
    assert partial['accepted_iterations'] == 1 and not partial['converged']
    calls = []
    resumed = optimize_member_lm(prior, center, target, variance, batch_of(evaluate, calls),
                                 checkpoint=path, identity='same', max_iterations=100)
    assert resumed['converged']
    assert resumed['history'][:2] == partial['history']
    assert calls and set(calls) == {4}          # no re-evaluation of the accepted point
    with pytest.raises(ValueError, match='different inputs'):
        optimize_member_lm(prior, center, target, variance, batch_of(evaluate),
                           checkpoint=path, identity='changed')
    with pytest.raises(ValueError, match='different optimizer'):
        optimize_member(prior, center, target, variance, evaluate, checkpoint=path, identity='same')


def test_lm_time_budget_keeps_last_accepted_state(monkeypatch, tmp_path):
    import rml_standalone
    clock = iter([0., 0., 5., 5.])
    monkeypatch.setattr(rml_standalone.time, 'monotonic', lambda: next(clock))
    matrix = np.ones((1, 8))
    fit = optimize_member_lm(prior_fixture(), np.zeros(8), np.ones(1), np.ones(1),
                             batch_of(lambda model: (matrix @ model, matrix)), max_seconds=1.)
    assert fit['reason'] == 'time_limit' and fit['evaluations_this_attempt'] == 1
    np.testing.assert_array_equal(fit['z'], np.zeros(8))


def test_factor_solve_inverts_factor_apply():
    from rml_standalone import factor_solve
    prior = prior_fixture()
    values = np.random.default_rng(2).normal(size=(8, 3))
    np.testing.assert_allclose(factor_solve(prior, factor_apply(prior, values)), values, atol=1e-10)
    np.testing.assert_allclose(factor_apply(prior, factor_solve(prior, values[:, 0])), values[:, 0], atol=1e-10)


def test_lm_from_a_distant_start_reaches_the_same_linear_gaussian_minimizer(tmp_path):
    prior = prior_fixture()
    rng = np.random.default_rng(28)
    matrix = rng.normal(size=(3, 8))
    observations, variance = np.array([.1, -.3, .2]), np.array([.02, .03, .04])
    inputs = randomized_inputs(prior, observations, variance, 2, 2045)
    covariance = prior.apply(np.eye(8))
    gain = covariance @ matrix.T @ np.linalg.inv(matrix @ covariance @ matrix.T + np.diag(variance))
    center, target = inputs['prior_centers'][0], inputs['targets'][0]
    start = center + rng.normal(size=8)
    seen = []
    def evaluate_batch(models):
        seen.extend(models)
        return [(matrix @ model, matrix) for model in models]
    fit = optimize_member_lm(prior, center, target, variance, evaluate_batch, start=start,
                             relative_gradient_tolerance=1e-10, stagnation_tolerance=1e-14,
                             max_iterations=300)
    np.testing.assert_allclose(seen[0], start, atol=1e-10)
    assert fit['history'][0]['prior_loss'] > 0
    np.testing.assert_allclose(fit['model'], center + gain @ (target - matrix @ center), atol=1e-6)


def test_forward_only_trials_match_eager_lm_and_reuse_final_jacobian():
    prior = prior_fixture()
    rng = np.random.default_rng(31)
    matrix = rng.normal(size=(3, 8))
    center, target, variance = rng.normal(size=8), np.array([.4, -.2, .9]), np.array([.02, .03, .04])
    def evaluate(model):
        values = matrix @ model
        return np.sin(values), np.cos(values)[:, None] * matrix
    eager = optimize_member_lm(prior, center, target, variance, batch_of(evaluate), max_iterations=8)
    jacobian_batches, forward_batches = [], []
    def forward(models):
        forward_batches.append(len(models))
        return [evaluate(model)[0] for model in models]
    lazy = optimize_member_lm(prior, center, target, variance, batch_of(evaluate, jacobian_batches),
                              max_iterations=8, evaluate_forward=forward)
    np.testing.assert_allclose(lazy['model'], eager['model'], atol=1e-12)
    np.testing.assert_allclose([row['objective'] for row in lazy['history']],
                               [row['objective'] for row in eager['history']], atol=1e-12)
    assert set(jacobian_batches) == {1} and len(jacobian_batches) == len(lazy['history'])
    assert set(forward_batches) == {4}
    np.testing.assert_allclose(lazy['jacobian'], evaluate(lazy['model'])[1])


def test_forward_only_lm_handles_failed_trials_and_failed_linearization():
    prior = prior_fixture()
    matrix = np.ones((1, 8))
    center, target, variance = np.zeros(8), np.ones(1), np.array([.1])
    calls = []
    def jacobian_batch(models):
        calls.append(len(models))
        if len(calls) == 2:
            return [None]
        return [(matrix @ model, matrix) for model in models]
    def forward(models):
        values = [matrix @ model for model in models]
        values[0] = np.full(1, np.nan)
        return values
    fit = optimize_member_lm(prior, center, target, variance, jacobian_batch,
                             evaluate_forward=forward, max_iterations=2)
    assert any(row['failed'] for row in fit['trial_history'])
    # The failed linearization is retried at a slightly shorter step instead of shrinking the radius.
    assert any(row.get('linearization_failures') == [1.] and row.get('linearization_scale') == .97
               for row in fit['trial_history'])
    assert fit['accepted_iterations'] >= 1
    assert np.all(np.diff([row['objective'] for row in fit['history']]) < 0)


def linear_gaussian_case(seed=28):
    prior = prior_fixture()
    matrix = np.random.default_rng(seed).normal(size=(3, 8))
    target, variance = np.array([.1, -.3, .2]), np.array([.02, .03, .04])
    covariance = prior.apply(np.eye(8))
    gain = covariance @ matrix.T @ np.linalg.inv(matrix @ covariance @ matrix.T + np.diag(variance))
    return prior, matrix, target, variance, gain @ target


def test_lm_failure_rounds_keep_trust_and_damping_and_solve_new_models_between_the_failed_ones():
    prior, matrix, target, variance, minimizer = linear_gaussian_case()
    calls = []
    def evaluate_batch(models):
        calls.append(len(models))
        if len(calls) == 2:                          # every candidate of the first round fails
            return [None] * len(models)
        return [(matrix @ model, matrix) for model in models]
    fit = optimize_member_lm(prior, np.zeros(8), target, variance, evaluate_batch, max_iterations=200,
                             relative_gradient_tolerance=1e-9, stagnation_tolerance=1e-12)
    first, second = fit['trial_history'][:4], fit['trial_history'][4:8]
    assert all(row['failed'] and row['failure_round'] for row in first)
    assert all(row['trust'] == .3 for row in first + second)
    assert [row['damping'] for row in second] == [row['damping'] for row in first]
    assert all(row['step_scale'] == pytest.approx(2 ** (-1 / 3)) for row in second)
    np.testing.assert_allclose([row['step_inf'] for row in second],
                               [2 ** (-1 / 3) * row['step_inf'] for row in first])
    # New models at the scale of the failed ones, not a shorter ladder.
    assert not {row['step_inf'] for row in second} & {row['step_inf'] for row in first}
    assert min(row['step_inf'] for row in first) < second[0]['step_inf'] < first[0]['step_inf']
    accepted = fit['history'][1]
    assert fit['converged'] and accepted['round'] == 2
    ratio, damping = accepted['gain_ratio'], first[accepted['candidate']]['damping']
    assert accepted['next_damping'] == pytest.approx(max(1e-6, damping * max(1 / 3, 1 - (2 * ratio - 1) ** 3)))
    np.testing.assert_allclose(fit['model'], minimizer, atol=1e-6)


def test_lm_stops_as_solver_failures_when_every_round_fails():
    prior = prior_fixture()
    calls = []
    def evaluate_batch(models):
        calls.append(len(models))
        if len(calls) == 1:
            return [(np.array([model.sum()]), np.ones((1, 8))) for model in models]
        return [None] * len(models)
    fit = optimize_member_lm(prior, np.zeros(8), np.ones(1), np.ones(1), evaluate_batch)
    assert fit['reason'] == 'solver_failures' and not fit['converged'] and fit['accepted_iterations'] == 0
    from pilot_rml_index45 import TERMINAL_REASONS
    assert fit['reason'] in TERMINAL_REASONS          # a finished member, not one to resubmit
    assert len(calls) == 5                           # the start, then 1 + len(FAILURE_SCALES) rounds
    trials = fit['trial_history']
    assert [row['trust'] for row in trials[::4]] == [.3] * 4
    assert [row.get('step_scale', 1.) for row in trials[::4]] == pytest.approx(
        [1., 2 ** (-1 / 3), 2 ** (-2 / 3), 1 / 16])
    sizes = [{row['step_inf'] for row in trials[start:start + 4]} for start in range(0, 16, 4)]
    assert all(sizes[k].isdisjoint(set().union(*sizes[:k])) for k in range(1, 4))  # new models only
    # The last retry continues the ladder below its smallest failed step.
    assert max(sizes[3]) < min(sizes[0])


def test_lm_retries_a_failed_start_from_a_reproducible_nearby_point():
    from rml_standalone import factor_solve
    prior = prior_fixture()
    matrix = np.ones((1, 8))
    def run():
        seen = []
        def evaluate_batch(models):
            seen.append(models[0].copy())
            return [None] if len(seen) == 1 else [(matrix @ model, matrix) for model in models]
        fit = optimize_member_lm(prior, np.zeros(8), np.ones(1), np.array([.1]), evaluate_batch,
                                 identity='start', max_iterations=3)
        return fit, seen
    fit, seen = run()
    assert fit['history'][0]['start_retries'] == 1 and fit['accepted_iterations'] >= 1
    offset = np.max(np.abs(factor_solve(prior, seen[1])))
    assert 0 < offset < 1e-2
    np.testing.assert_array_equal(run()[1][1], seen[1])


def test_lm_falls_back_to_the_prior_centre_when_a_supplied_start_cannot_be_solved():
    prior, matrix, target, variance, minimizer = linear_gaussian_case()
    start = np.full(8, 3.)
    seen = []
    def evaluate_batch(models):
        seen.extend(np.asarray(m) for m in models)
        # Every model near the supplied start fails; everything else solves.
        return [None if np.linalg.norm(model - start) < 1. else (matrix @ model, matrix) for model in models]
    fit = optimize_member_lm(prior, np.zeros(8), target, variance, evaluate_batch, start=start, identity='warm',
                             max_iterations=200, relative_gradient_tolerance=1e-9, stagnation_tolerance=1e-12)
    assert fit['history'][0]['start_retries'] == 3 and fit['history'][0]['start_fallback'] == 'prior centre'
    np.testing.assert_allclose(seen[4], np.zeros(8), atol=1e-12)        # start, 3 retries, then the centre
    assert fit['converged']
    np.testing.assert_allclose(fit['model'], minimizer, atol=1e-6)


def test_lm_retries_a_failed_linearization_with_a_slightly_shorter_step():
    prior, matrix, target, variance, _ = linear_gaussian_case(31)
    linearized = []
    def jacobian_batch(models):
        linearized.append(models[0].copy())
        if len(linearized) == 2:                     # the accepted candidate's first linearization
            return [None]
        return [(matrix @ model, matrix) for model in models]
    fit = optimize_member_lm(prior, np.zeros(8), target, variance, jacobian_batch,
                             evaluate_forward=lambda models: [matrix @ model for model in models],
                             max_iterations=1)
    retried = [row for row in fit['trial_history'] if row.get('linearization_failures')]
    assert len(retried) == 1 and retried[0]['linearization_scale'] == .97
    np.testing.assert_allclose(linearized[2], .97 * linearized[1])
    assert fit['history'][1]['step_inf'] == pytest.approx(.97 * retried[0]['step_inf'])
    assert fit['accepted_iterations'] == 1


def test_forward_only_lm_rejects_a_trial_if_relinearization_worsens_objective():
    prior = prior_fixture()
    matrix = np.ones((1, 8))
    def evaluate(models):
        return [(np.array([0. if not np.any(model) else -1.]), matrix) for model in models]
    def forward(models):
        return [np.array([.9]) for _ in models]
    fit = optimize_member_lm(prior, np.zeros(8), np.ones(1), np.array([.1]), evaluate,
                             evaluate_forward=forward, max_rounds=1)
    assert fit['reason'] == 'nonlinearity_limited' and fit['accepted_iterations'] == 0
    assert any(row.get('linearization_rejected') for row in fit['trial_history'])
    np.testing.assert_array_equal(fit['model'], np.zeros(8))


def scheduled_case():
    prior = prior_fixture()
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(3, 8))
    target, variance = matrix @ np.full(8, 2.), np.full(3, 1e-2)
    holder, used = {'tolerance': None}, []
    return prior, matrix, target, variance, holder, used


def test_lm_tolerance_schedule_tightens_with_the_misfit_and_reevaluates_the_incumbent():
    prior, matrix, target, variance, holder, used = scheduled_case()
    def evaluate_batch(models):
        used.append(holder['tolerance'])
        return [(matrix @ model, matrix) for model in models]
    schedule = [[20., 1e-2], [1., 3e-3], [0., 1e-3]]
    fit = optimize_member_lm(prior, np.zeros(8), target, variance, evaluate_batch, max_iterations=100,
                             relative_gradient_tolerance=1e-9, stagnation_tolerance=1e-12,
                             tolerance_schedule=schedule,
                             set_tolerance=lambda value: holder.__setitem__('tolerance', value))
    assert used[0] == 1e-2 and fit['converged'] and fit['solver_tolerance'] == 1e-3
    tolerances = [row['solver_tolerance'] for row in fit['history']]
    assert tolerances[0] == 1e-2 and tolerances[-1] == 1e-3 and np.all(np.diff(tolerances) <= 0)
    misfits = [2 * row['likelihood_loss'] / 3 for row in fit['history']]
    assert misfits[0] > 20
    # The tolerance recorded with each accepted iteration is the one its evaluations used,
    # and the incumbent was re-evaluated at the tighter tolerance once its misfit crossed a threshold.
    refined = [row for row in fit['history'] if 'refined_tolerance' in row]
    assert [row['refined_tolerance'] for row in refined] == [3e-3, 1e-3]
    for row in refined:
        assert row['refined_objective'] == pytest.approx(row['objective'])      # the fake solver is exact
        assert 2 * row['likelihood_loss'] / 3 <= (20. if row['refined_tolerance'] == 3e-3 else 1.)
    following = fit['history'][fit['history'].index(refined[0]) + 1]
    assert following['solver_tolerance'] == 3e-3
    assert all(row['solver_tolerance'] in (1e-2, 3e-3, 1e-3) for row in fit['trial_history'])
    assert used[-1] == 1e-3


def test_lm_keeps_the_looser_tolerance_when_the_refinement_fails():
    prior, matrix, target, variance, holder, used = scheduled_case()
    def evaluate_batch(models):
        used.append(holder['tolerance'])
        return [None if holder['tolerance'] == 3e-3 else (matrix @ model, matrix) for model in models]
    fit = optimize_member_lm(prior, np.zeros(8), target, variance, evaluate_batch, max_iterations=100,
                             relative_gradient_tolerance=1e-9, stagnation_tolerance=1e-12,
                             tolerance_schedule=[[20., 1e-2], [1., 3e-3], [0., 1e-3]],
                             set_tolerance=lambda value: holder.__setitem__('tolerance', value))
    retained = [row for row in fit['history'] if 'tolerance_retained' in row]
    assert len(retained) == 1 and retained[0]['tolerance_retained'] == 3e-3      # blocked after one failure
    assert not any(row.get('refined_tolerance') == 3e-3 for row in fit['history'])
    assert fit['converged'] and fit['solver_tolerance'] == 1e-3 and fit['blocked_tolerances'] == [3e-3]
    assert 3e-3 not in [row['solver_tolerance'] for row in fit['history']]
    assert used.count(3e-3) == 1


def test_lm_backs_off_the_tolerance_when_a_whole_round_stalls():
    prior, matrix, target, variance, holder, used = scheduled_case()
    def evaluate_batch(models):
        used.append(holder['tolerance'])
        # The refinement at 1e-3 (a single-point batch) solves; every later 1e-3 batch stalls.
        stall = holder['tolerance'] == 1e-3 and used.count(1e-3) > 1
        return [None if stall else (matrix @ model, matrix) for model in models]
    fit = optimize_member_lm(prior, np.zeros(8), target, variance, evaluate_batch, max_iterations=100,
                             relative_gradient_tolerance=1e-9, stagnation_tolerance=1e-12,
                             tolerance_schedule=[[20., 1e-2], [1., 3e-3], [0., 1e-3]],
                             set_tolerance=lambda value: holder.__setitem__('tolerance', value))
    backoff = [row for row in fit['trial_history'] if 'tolerance_backoff' in row]
    assert backoff and all(row['tolerance_backoff'] == 3e-3 and row['failed'] for row in backoff)
    assert not any(row.get('failure_round') for row in fit['trial_history'])
    assert fit['converged'] and fit['solver_tolerance'] == 3e-3 and fit['blocked_tolerances'] == [1e-3]
    assert used.count(1e-3) == 2                       # the refinement and the one stalled round
    assert fit['history'][-1]['solver_tolerance'] == 3e-3


def test_lm_rejects_malformed_tolerance_schedules():
    prior, matrix, target, variance, holder, used = scheduled_case()
    evaluate_batch = batch_of(lambda model: (matrix @ model, matrix))
    setter = lambda value: None
    for schedule in ([[1., 1e-2], [20., 1e-3]], [[20., 1e-3], [0., 1e-2]], [[20., 1.]], [], [[20.]]):
        with pytest.raises(ValueError):
            optimize_member_lm(prior, np.zeros(8), target, variance, evaluate_batch,
                               tolerance_schedule=schedule, set_tolerance=setter)
    with pytest.raises(ValueError, match='set_tolerance'):
        optimize_member_lm(prior, np.zeros(8), target, variance, evaluate_batch, tolerance_schedule=[[0., 1e-3]])
