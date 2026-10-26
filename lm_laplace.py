"""Nonlinear final-stage MAP fitting and data-space Laplace sampling."""

import time
from copy import copy

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class SpatialPrior:
    """Positive-definite separable Gaussian covariance with posterior-scaled spread.

    This is a structured approximation to the incoming ensemble covariance,
    not its rank-deficient sample covariance. The explicit standard-deviation
    floor retains uncertainty outside the ensemble span.
    """

    def __init__(self, ensemble, shape, spacing, lengths, std_floor=(.23, .1)):
        self.mean = ensemble.mean(axis=1)
        self.shape = (2, *shape)
        self.std = np.maximum(ensemble.std(axis=1, ddof=1),
                              np.repeat(std_floor, int(np.prod(shape))))
        if np.ndim(lengths) == 1:
            per_block = [list(lengths)] * self.shape[0]
        else:
            per_block = [list(block) for block in lengths]
        if len(per_block) != self.shape[0] or any(len(block) != len(shape) for block in per_block):
            raise ValueError('Correlation lengths must provide one value per block and axis')
        self.factors = []
        for block in per_block:
            block_factors = []
            for size, step, length in zip(shape, spacing, block):
                distance = abs(np.arange(size)[:, None] - np.arange(size)[None, :]) * step
                block_factors.append(np.linalg.cholesky(np.exp(-distance / length)))
            self.factors.append(block_factors)

    def _axis(self, values, axis, operation):
        moved = np.moveaxis(values, axis, 0)
        result = operation(moved.reshape(moved.shape[0], -1))
        return np.moveaxis(result.reshape(moved.shape), 0, axis)

    def _apply_block(self, block, factors, operation):
        for axis, factor in enumerate(factors):
            block = self._axis(block, axis, lambda x, f=factor: operation(x, f))
        return block

    def apply(self, values):
        vector = np.asarray(values).ndim == 1
        values = np.asarray(values).reshape(len(self.mean), -1)
        fields = (self.std[:, None] * values).reshape(*self.shape, -1)
        for block_idx, factors in enumerate(self.factors):
            fields[block_idx] = self._apply_block(
                fields[block_idx], factors, lambda x, f: f @ (f.T @ x))
        result = self.std[:, None] * fields.reshape(len(self.mean), -1)
        return result[:, 0] if vector else result

    def solve(self, values):
        vector = np.asarray(values).ndim == 1
        values = np.asarray(values).reshape(len(self.mean), -1)
        fields = (values / self.std[:, None]).reshape(*self.shape, -1)
        for block_idx, factors in enumerate(self.factors):
            fields[block_idx] = self._apply_block(
                fields[block_idx], factors, lambda x, f: cho_solve((f, True), x))
        result = fields.reshape(len(self.mean), -1) / self.std[:, None]
        return result[:, 0] if vector else result

    def draw(self, count, rng):
        fields = rng.normal(size=(*self.shape, count))
        for block_idx, factors in enumerate(self.factors):
            fields[block_idx] = self._apply_block(
                fields[block_idx], factors, lambda x, f: f @ x)
        return self.std[:, None] * fields.reshape(len(self.mean), count)


def map_objective(model, prediction, prior, datum, variance):
    offset = model - prior.mean
    penalty = (prior.quadratic(offset) if hasattr(prior, 'quadratic')
               else offset @ prior.solve(offset))
    return float(.5 * (np.sum((datum - prediction) ** 2 / variance)
                       + penalty))


def optimize(initial, prior, datum, variance, forward, derivative, max_iterations=12,
              max_trials=5, max_seconds=1200., max_step=.25, relative_data_improvement=.05,
              initial_prediction=None, rh_limits=(-1., 4.), step_weights=None):
    """Prior-preconditioned LM; accept only actual nonlinear objective decreases."""
    if not np.isfinite(relative_data_improvement) or not 0 < relative_data_improvement < 1:
        raise ValueError('Relative data improvement threshold must lie between zero and one')
    if (rh_limits is not None and (len(rh_limits) != 2 or not np.isfinite(rh_limits).all()
                                   or rh_limits[0] >= rh_limits[1])):
        raise ValueError('Rh bounds must be finite and increasing')
    started = time.monotonic()
    if step_weights is not None:
        step_weights = np.asarray(step_weights, dtype=float)
        if (step_weights.shape != initial.shape or not np.isfinite(step_weights).all()
                or np.any((step_weights < 0) | (step_weights > 1))):
            raise ValueError('Step weights must match the model and lie in [0, 1]')
    model = initial.copy()
    prediction = forward(model) if initial_prediction is None else initial_prediction
    objective = map_objective(model, prediction, prior, datum, variance)
    history = [objective]
    data_misfit = float(np.mean((datum - prediction) ** 2 / variance))
    data_history = [data_misfit]
    relative_history = []
    damping = 1.
    errors = []
    converged, reason = False, 'iteration_limit'
    jacobian = None
    for iteration in range(max_iterations):
        if time.monotonic() - started >= max_seconds:
            reason = 'time_limit'
            break
        jacobian = derivative(model)
        cj = prior.apply(jacobian.T)
        gram = jacobian @ cj
        offset = model - prior.mean
        accepted = False
        for _ in range(max_trials):
            if time.monotonic() - started >= max_seconds:
                reason = 'time_limit'
                break
            matrix = gram + np.diag((1. + damping) * variance)
            rhs = datum - prediction + jacobian @ offset / (1. + damping)
            step = cj @ np.linalg.solve(matrix, rhs) - offset / (1. + damping)
            if step_weights is not None:
                step *= step_weights
            step *= min(1., max_step / max(np.max(abs(step)), 1e-15))
            candidate = model + step
            cells = len(model) // 2
            if rh_limits is not None:
                candidate[:cells] = np.clip(candidate[:cells], *rh_limits)
            candidate[cells:] = np.clip(candidate[cells:], -4., 4.)
            if step_weights is not None:
                candidate[step_weights == 0] = initial[step_weights == 0]
            try:
                trial_prediction = forward(candidate)
                trial_objective = map_objective(candidate, trial_prediction, prior, datum, variance)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                errors.append(str(exc))
                trial_objective = np.inf
            if np.isfinite(trial_objective) and trial_objective < objective:
                trial_data_misfit = float(np.mean((datum - trial_prediction) ** 2 / variance))
                relative_improvement = ((data_misfit - trial_data_misfit)
                                        / max(data_misfit, np.finfo(float).tiny))
                model, prediction, objective = candidate, trial_prediction, trial_objective
                history.append(objective)
                data_misfit = trial_data_misfit
                data_history.append(data_misfit)
                relative_history.append(relative_improvement)
                damping = max(damping / 3., 1e-8)
                accepted = True
                if 0 <= relative_improvement < relative_data_improvement:
                    converged, reason = True, 'relative_data_improvement'
                break
            damping *= 10.
        if converged:
            break
        if not accepted:
            reason = 'time_limit' if reason == 'time_limit' else 'no_acceptable_trial'
            break
    # Curvature must correspond to the saved solution, not the preceding iterate.
    jacobian = derivative(model)
    return {'model': model, 'prediction': prediction, 'jacobian': jacobian,
            'objective_history': history, 'converged': converged, 'reason': reason,
            'data_misfit_history': data_history,
            'relative_data_improvement_history': relative_history,
            'trial_errors': errors, 'seconds': time.monotonic() - started}


def optimize_batch(initials, initial_predictions, prior, datum, variance,
                   forward_batch, derivative_batch, max_iterations=12, max_trials=5,
                   max_seconds=1200., max_step=.25, relative_data_improvement=.05,
                   classify_failure=None, prior_means=None, datums=None,
                   rh_limits=(-1., 4.), final_jacobian=True, progress=None):
    """Run independent LM chains with batched 2D forward and Jacobian calls."""
    if len(initials) != len(initial_predictions):
        raise ValueError('Each LM start must have one initial prediction')
    if not 0 < relative_data_improvement < 1:
        raise ValueError('Relative data improvement threshold must lie between zero and one')
    if prior_means is not None and np.shape(prior_means) != (len(initials), len(prior.mean)):
        raise ValueError('Each optimizer needs one prior center')
    if datums is not None and np.shape(datums) != (len(initials), len(datum)):
        raise ValueError('Each optimizer needs one observation target')
    started = time.monotonic()
    states = []
    for member, (initial, prediction) in enumerate(zip(initials, initial_predictions)):
        member_prior = copy(prior)
        member_prior.mean = prior.mean if prior_means is None else np.asarray(prior_means[member])
        target = datum if datums is None else np.asarray(datums[member])
        model = np.asarray(initial, dtype=float).copy()
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != np.shape(datum) or not np.isfinite(prediction).all():
            states.append({'model': model, 'prediction': prediction, 'converged': False,
                           'reason': 'invalid_initial_prediction', 'failure_stage': 'initial_forward',
                           'trial_errors': [], 'objective_history': [],
                           'data_misfit_history': [], 'relative_data_improvement_history': []})
            continue
        objective = map_objective(model, prediction, member_prior, target, variance)
        if not np.isfinite(objective):
            states.append({'model': model, 'prediction': prediction, 'converged': False,
                           'reason': 'invalid_initial_objective', 'failure_stage': 'initial_objective',
                           'trial_errors': [], 'objective_history': [],
                           'data_misfit_history': [], 'relative_data_improvement_history': []})
            continue
        states.append({'model': model, 'prediction': prediction, 'objective': objective,
                       'prior': member_prior, 'target': target,
                       'data_misfit': float(np.mean((target - prediction) ** 2 / variance)),
                       'converged': False, 'reason': 'iteration_limit', 'damping': 1.,
                       'objective_history': [objective],
                       'data_misfit_history': [float(np.mean((target - prediction) ** 2 / variance))],
                       'relative_data_improvement_history': [], 'trial_errors': [],
                       'seconds': 0.})

    for iteration in range(max_iterations):
        active = [state for state in states
                  if state['reason'] == 'iteration_limit' and not state['converged']]
        for state in active:
            state['seconds'] = time.monotonic() - started
            if state['seconds'] >= max_seconds:
                state['reason'] = 'time_limit'
        active = [state for state in active if state['reason'] == 'iteration_limit']
        if not active:
            break
        if progress is not None:
            progress(iteration + 1, len(active))

        jacobians = derivative_batch([state['model'] for state in active])
        if len(jacobians) != len(active):
            raise ValueError('Batched Jacobian count does not match active LM starts')
        for state, jacobian in zip(active, jacobians):
            if isinstance(jacobian, Exception) or jacobian is None:
                state['reason'] = 'jacobian_failure'
                state['failure_stage'] = ('initial_jacobian' if iteration == 0
                                          else 'optimizer_jacobian')
                state['failure'] = (str(jacobian) if isinstance(jacobian, Exception)
                                    else 'Batched Jacobian was unavailable')
                if isinstance(jacobian, Exception):
                    state['failure_error'] = jacobian
                continue
            jacobian = np.asarray(jacobian, dtype=float)
            if jacobian.shape != (len(datum), len(state['model'])) or not np.isfinite(jacobian).all():
                state['reason'] = 'invalid_jacobian'
                state['failure_stage'] = ('initial_jacobian' if iteration == 0
                                          else 'optimizer_jacobian')
                state['failure'] = 'Batched Jacobian had an invalid shape or non-finite entries'
                continue
            state['jacobian'] = jacobian
            state['cj'] = prior.apply(jacobian.T)
            state['gram'] = jacobian @ state['cj']
            state['offset'] = state['model'] - state['prior'].mean
            state['accepted_this_iteration'] = False

        for _ in range(max_trials):
            pending = [state for state in active
                       if state['reason'] == 'iteration_limit'
                       and not state['accepted_this_iteration']]
            for state in pending:
                state['seconds'] = time.monotonic() - started
                if state['seconds'] >= max_seconds:
                    state['reason'] = 'time_limit'
            pending = [state for state in pending if state['reason'] == 'iteration_limit']
            if not pending:
                break

            trial_models = []
            for state in pending:
                jacobian = state['jacobian']
                matrix = state['gram'] + np.diag((1. + state['damping']) * variance)
                rhs = (state['target'] - state['prediction']
                       + jacobian @ state['offset'] / (1. + state['damping']))
                step = (state['cj'] @ np.linalg.solve(matrix, rhs)
                        - state['offset'] / (1. + state['damping']))
                step *= min(1., max_step / max(np.max(abs(step)), 1e-15))
                candidate = state['model'] + step
                cells = len(candidate) // 2
                candidate[:cells] = np.clip(candidate[:cells], *rh_limits)
                candidate[cells:] = np.clip(candidate[cells:], -4., 4.)
                trial_models.append(candidate)

            trial_predictions = forward_batch(trial_models)
            if len(trial_predictions) != len(pending):
                raise ValueError('Batched forward count does not match active LM trials')
            for state, candidate, trial_prediction in zip(pending, trial_models, trial_predictions):
                if isinstance(trial_prediction, Exception) or trial_prediction is None:
                    state['trial_errors'].append(
                        str(trial_prediction) if isinstance(trial_prediction, Exception)
                        else 'Batched forward returned no valid response')
                    state['damping'] *= 10.
                    continue
                trial_prediction = np.asarray(trial_prediction, dtype=float)
                if trial_prediction.shape != np.shape(datum) or not np.isfinite(trial_prediction).all():
                    state['trial_errors'].append('Batched forward returned an invalid prediction')
                    state['damping'] *= 10.
                    continue
                trial_objective = map_objective(candidate, trial_prediction, state['prior'],
                                                state['target'], variance)
                if np.isfinite(trial_objective) and trial_objective < state['objective']:
                    trial_data_misfit = float(np.mean((state['target'] - trial_prediction) ** 2 / variance))
                    relative_improvement = ((state['data_misfit'] - trial_data_misfit)
                                            / max(state['data_misfit'], np.finfo(float).tiny))
                    state.update(model=candidate, prediction=trial_prediction,
                                 objective=trial_objective, data_misfit=trial_data_misfit,
                                 accepted_this_iteration=True)
                    state['objective_history'].append(trial_objective)
                    state['data_misfit_history'].append(trial_data_misfit)
                    state['relative_data_improvement_history'].append(relative_improvement)
                    state['damping'] = max(state['damping'] / 3., 1e-8)
                    if 0 <= relative_improvement < relative_data_improvement:
                        state['converged'] = True
                        state['reason'] = 'relative_data_improvement'
                else:
                    state['damping'] *= 10.

        for state in active:
            if (state['reason'] == 'iteration_limit'
                    and not state['converged']
                    and not state.get('accepted_this_iteration', False)):
                state['reason'] = 'no_acceptable_trial'
            state['seconds'] = time.monotonic() - started

    converged = [state for state in states if state['converged']]
    if converged and final_jacobian:
        jacobians = derivative_batch([state['model'] for state in converged])
        if len(jacobians) != len(converged):
            raise ValueError('Final batched Jacobian count does not match converged LM starts')
        for state, jacobian in zip(converged, jacobians):
            if isinstance(jacobian, Exception) or jacobian is None:
                state['converged'] = False
                state['reason'] = 'final_jacobian_failure'
                state['failure_stage'] = 'final_jacobian'
                state['failure'] = (str(jacobian) if isinstance(jacobian, Exception)
                                    else 'Final batched Jacobian was unavailable')
                if isinstance(jacobian, Exception):
                    state['failure_error'] = jacobian
                continue
            jacobian = np.asarray(jacobian, dtype=float)
            if jacobian.shape != (len(datum), len(state['model'])) or not np.isfinite(jacobian).all():
                state['converged'] = False
                state['reason'] = 'invalid_final_jacobian'
                state['failure_stage'] = 'final_jacobian'
                state['failure'] = 'Final batched Jacobian had an invalid shape or non-finite entries'
            else:
                state['jacobian'] = jacobian

    for state in states:
        state['seconds'] = time.monotonic() - started
        state.pop('prior', None)
        state.pop('target', None)
    return states


def sample_laplace(solution, prior, variance, count, rng):
    """Conditional simulation of (C^-1 + J' R^-1 J)^-1; no LM damping."""
    jacobian = solution['jacobian']
    cj = prior.apply(jacobian.T)
    factor = cho_factor(jacobian @ cj + np.diag(variance), lower=True)
    deviations = prior.draw(count, rng)
    noise = rng.normal(size=(len(variance), count)) * np.sqrt(variance)[:, None]
    deviations -= cj @ cho_solve(factor, jacobian @ deviations + noise)
    return solution['model'][:, None] + deviations


def final_stage(ensemble, predictions, datum, original_variance, inflation, shape, spacing,
                lengths, threshold, forward_1d, jacobian_1d, forward_2d, jacobian_2d,
                rng, max_iterations=12, max_seconds=1200., max_step=.25,
                relative_data_improvement=.05, is_recoverable_2d_failure=None,
                invariant_start=None, max_2d_starts=5,
                forward_2d_batch=None, jacobian_2d_batch=None):
    if int(max_2d_starts) != max_2d_starts or max_2d_starts < 1:
        raise ValueError('Require at least one 2D candidate start')
    scores = np.mean((predictions - datum[:, None]) ** 2 / original_variance[:, None], axis=0)
    best = int(np.argmin(scores))
    prior = SpatialPrior(ensemble, shape, spacing, lengths)
    variance = inflation * original_variance
    options = dict(max_iterations=max_iterations, max_trials=5,
                   max_seconds=max_seconds, max_step=max_step,
                   relative_data_improvement=relative_data_improvement)
    fits = []
    gate_fidelity = '1D' if scores[best] <= threshold else '2D'
    failures = []
    candidate_fits = []
    expected_2d_starts = 0

    def retry_reason(exc):
        if is_recoverable_2d_failure is None:
            return None
        reason = is_recoverable_2d_failure(exc)
        return 'exit_174' if reason is True else reason

    def record_failure(member, kind, reason, error, stage, errors=None):
        failure = {'member': member, 'start_kind': kind, 'reason': reason,
                   'stage': stage, 'error': error}
        if reason in ('exit_174', 'exit_135', 'exit_139'):
            failure['exit_status'] = int(reason.split('_')[1])
        if errors is not None:
            failure['trial_errors'] = errors
        failures.append(failure)
        fits.append({'fidelity': '2D', 'member': member, 'start_kind': kind,
                     'reason': f'{stage}_{reason}', 'error': error})

    def forward_batch(models):
        if forward_2d_batch is not None:
            result = list(forward_2d_batch(models))
        else:
            result = []
            for model in models:
                try:
                    result.append(forward_2d(model))
                except RuntimeError as exc:
                    result.append(exc)
        if len(result) != len(models):
            raise ValueError('Batched 2D forward count does not match candidate count')
        return result

    def derivative_batch(models):
        if jacobian_2d_batch is not None:
            result = list(jacobian_2d_batch(models))
        else:
            result = []
            for model in models:
                try:
                    result.append(jacobian_2d(model))
                except RuntimeError as exc:
                    result.append(exc)
        if len(result) != len(models):
            raise ValueError('Batched 2D Jacobian count does not match candidate count')
        return result

    if gate_fidelity == '2D':
        ranked = [int(c) for c in np.argsort(scores, kind='stable')[:max_2d_starts]]
        queue = [(member, 'ranked_member', ensemble[:, member]) for member in ranked]
        if invariant_start is not None:
            queue.insert(0, (ranked[0], 'invariant_extension',
                             invariant_start(ensemble[:, ranked[0]])))
        expected_2d_starts = len(queue)
        initial_predictions = forward_batch([item[2] for item in queue])
        viable_starts, viable_predictions = [], []
        for (member, kind, initial), prediction in zip(queue, initial_predictions):
            if isinstance(prediction, Exception):
                reason = retry_reason(prediction)
                if not reason:
                    raise prediction
                record_failure(member, kind, reason, str(prediction), 'initial_forward')
                continue
            if prediction is None or not np.isfinite(np.asarray(prediction)).all():
                record_failure(member, kind, 'batch_forward_failure',
                               'No finite response returned for this start', 'initial_forward')
                continue
            viable_starts.append((member, kind, initial))
            viable_predictions.append(np.asarray(prediction, dtype=float))

        if viable_starts:
            optimized = optimize_batch(
                [item[2] for item in viable_starts], viable_predictions, prior, datum, variance,
                forward_batch, derivative_batch, max_iterations=options['max_iterations'],
                max_trials=options['max_trials'], max_seconds=options['max_seconds'],
                max_step=options['max_step'],
                relative_data_improvement=options['relative_data_improvement'])
            for (member, kind, _), fit in zip(viable_starts, optimized):
                if fit.get('failure') is not None:
                    failure_error = fit.get('failure_error')
                    reason = retry_reason(failure_error) if failure_error is not None else None
                    if failure_error is not None and not reason:
                        raise failure_error
                    record_failure(member, kind, reason or fit['reason'], fit['failure'],
                                   fit.get('failure_stage', 'optimizer'))
                    continue
                final_misfit = float(np.mean((fit['prediction'] - datum) ** 2
                                             / original_variance))
                candidate_fits.append({'fit': fit, 'member': member,
                                       'start_kind': kind, 'final_misfit': final_misfit})
                fits.append({'fidelity': '2D', 'member': member, 'start_kind': kind,
                             'converged': fit['converged'], 'reason': fit['reason'],
                             'history': fit['objective_history'],
                             'data_misfit_history': fit['data_misfit_history'],
                             'relative_data_improvement_history':
                                 fit['relative_data_improvement_history'],
                             'final_data_misfit': final_misfit,
                             'trial_errors': fit['trial_errors'], 'seconds': fit['seconds']})

    converged_candidates = [item for item in candidate_fits if item['fit']['converged']]
    has_converged_2d_start = gate_fidelity == '2D' and bool(converged_candidates)
    if has_converged_2d_start:
        selected = min(converged_candidates, key=lambda item: item['final_misfit'])
        fit = selected['fit']
        start_member, start_kind = selected['member'], selected['start_kind']
        fidelity = '2D'
    else:
        fidelity = '1D'
        start_member = best
        start_kind = ('1d_fallback' if gate_fidelity == '2D' else 'ranked_member')
        fit = optimize(ensemble[:, best], prior, datum, variance,
                       forward_1d, jacobian_1d, **options)
        fits.append({'fidelity': '1D', 'member': start_member, 'start_kind': start_kind,
                     'converged': fit['converged'], 'reason': fit['reason'],
                     'history': fit['objective_history'],
                     'data_misfit_history': fit['data_misfit_history'],
                     'relative_data_improvement_history':
                         fit['relative_data_improvement_history'],
                     'seconds': fit['seconds']})

    draws = sample_laplace(fit, prior, variance, ensemble.shape[1], rng)
    cells = len(fit['model']) // 2
    clipped = draws.copy()
    clipped[:cells] = np.clip(clipped[:cells], -1., 4.)
    clipped[cells:] = np.clip(clipped[cells:], -4., 4.)
    if fidelity == '1D':
        regenerated_predictions = np.column_stack([forward_1d(m) for m in clipped.T])
        kind = 'forward'
    else:
        regenerated_predictions = (fit['prediction'][:, None]
                                  + fit['jacobian'] @ (clipped - fit['model'][:, None]))
        kind = 'linearized'
    all_initial_failures_are_exit_174 = (
        bool(failures) and len(failures) == expected_2d_starts
        and all(item['reason'] == 'exit_174' and item['stage'] == 'initial_forward'
                for item in failures))
    fallback_reason = None
    if gate_fidelity == '2D' and fidelity == '1D':
        fallback_reason = ('2d_starts_exit_174' if all_initial_failures_are_exit_174
                           else 'no_converged_2d_start')
    return clipped, {
        'map_model': fit['model'], 'map_prediction': fit['prediction'],
        'map_forward_misfit': float(np.mean((fit['prediction'] - datum) ** 2 / original_variance)),
        'map_converged': fit['converged'], 'map_termination': fit['reason'],
        'convergence_relative_data_improvement': relative_data_improvement,
        'laplace_at_converged_map': fit['converged'], 'final_stage_history': fits,
        'best_member': best, 'best_member_data_misfit': float(scores[best]),
        'map_start_member': start_member, 'map_start_kind': start_kind,
        'gate_selected_fidelity': gate_fidelity,
        'two_d_initial_failures': failures,
        'invariant_start_attempted': bool(invariant_start is not None),
        'two_d_candidate_count': len(candidate_fits),
        'two_d_converged_candidate_count': len(converged_candidates),
        'required_converged_2d_starts': 1,
        'two_d_start_count': expected_2d_starts,
        'two_d_candidates': [{key: value for key, value in item.items() if key != 'fit'}
                             for item in candidate_fits],
        'fallback_reason': fallback_reason,
        'posterior_params': list(clipped.T), 'posterior_predictions': list(regenerated_predictions.T),
        'posterior_misfit': list(np.mean((regenerated_predictions - datum[:, None]) ** 2
                                        / original_variance[:, None], axis=0)),
        'posterior_prediction_kind': kind, 'posterior_misfit_kind': kind,
        'posterior_forward_verified': kind == 'forward', 'selected_fidelity': fidelity,
        'two_d_mode': ('lm-laplace' if fidelity == '2D' else
                       'abandoned' if failures else None),
        'prior_mean': prior.mean, 'prior_covariance': prior.std ** 2,
        'laplace_prior_kind': 'posterior_scaled_separable_exponential',
        'laplace_std_floor_ln': [.23, .1],
        'laplace_correlation_ft': list(lengths),
        'clipped_sample_fraction': float(np.mean(clipped != draws)),
        'trial_errors': fit['trial_errors'], 'optimized_2d': fidelity == '2D',
    }
