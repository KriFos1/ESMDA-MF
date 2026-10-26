"""Whitened, independently restartable randomized-posterior optimization."""

import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve


OPTIMIZER = 'whitened_damped_gauss_newton_v1'


def factor_apply(prior, values, transpose=False):
    """Apply L or L.T for the structured covariance C = L L.T."""
    if hasattr(prior, 'color'):                  # e.g. MaternPrior (sparse SPDE factor)
        return prior.color(values, transpose=transpose)
    values = np.asarray(values, dtype=float)
    vector = values.ndim == 1
    fields = values.reshape(*prior.shape, -1).copy()
    if transpose:
        fields *= prior.std.reshape(prior.shape)[..., None]
    for block, factors in enumerate(prior.factors):
        fields[block] = prior._apply_block(
            fields[block], factors, lambda x, f: (f.T if transpose else f) @ x)
    result = fields.reshape(len(prior.mean), -1)
    if not transpose:
        result *= prior.std[:, None]
    return result[:, 0] if vector else result


def factor_solve(prior, values):
    """Apply L^-1 for C = L L.T: whitened coordinates of a model offset."""
    if hasattr(prior, 'whiten'):
        return prior.whiten(values)
    from scipy.linalg import solve_triangular
    values = np.asarray(values, dtype=float)
    vector = values.ndim == 1
    fields = (values.reshape(len(prior.mean), -1) / prior.std[:, None]).reshape(*prior.shape, -1).copy()
    for block, factors in enumerate(prior.factors):
        fields[block] = prior._apply_block(
            fields[block], factors, lambda x, f: solve_triangular(f, x, lower=True))
    result = fields.reshape(len(prior.mean), -1)
    return result[:, 0] if vector else result


def randomized_inputs(prior, observations, variance, count, seed):
    observations, variance = np.asarray(observations), np.asarray(variance)
    if (count < 1 or observations.ndim != 1 or variance.shape != observations.shape
            or not np.isfinite(observations).all() or not np.isfinite(variance).all()
            or np.any(variance <= 0)):
        raise ValueError('Require positive count and finite data with positive variances')
    streams = np.random.SeedSequence(seed).spawn(count)
    centers, noises = [], []
    for stream in streams:
        prior_stream, data_stream = stream.spawn(2)
        centers.append(prior.mean + factor_apply(
            prior, np.random.default_rng(prior_stream).normal(size=len(prior.mean))))
        noises.append(np.random.default_rng(data_stream).normal(size=len(observations))
                      * np.sqrt(variance))
    centers, noises = np.asarray(centers), np.asarray(noises)
    return {'prior_centers': centers, 'measurement_noise': noises,
            'targets': observations[None, :] + noises,
            'observations': observations, 'variance': variance}


def objective_gradient(z, center, target, variance, prior, evaluate):
    return linearize(z, center, target, variance, prior, evaluate)[:5]


def linearize(z, center, target, variance, prior, evaluate):
    model = center + factor_apply(prior, z)
    prediction, jacobian = evaluate(model)
    return state_from(z, prediction, jacobian, target, variance, prior)


def state_from(z, prediction, jacobian, target, variance, prior):
    """Whitened randomized-posterior loss, gradient and sensitivity at an evaluated point."""
    prediction, jacobian = np.asarray(prediction), np.asarray(jacobian)
    if (prediction.shape != target.shape or jacobian.shape != (len(target), len(z))
            or not np.isfinite(prediction).all() or not np.isfinite(jacobian).all()):
        raise ValueError('Invalid prediction or Jacobian')
    residual = prediction - target
    prior_loss = .5 * float(z @ z)
    likelihood_loss = .5 * float(np.sum(residual ** 2 / variance))
    sensitivity = factor_apply(prior, jacobian.T, transpose=True).T / np.sqrt(variance)[:, None]
    gradient = z + sensitivity.T @ (residual / np.sqrt(variance))
    if not np.isfinite(gradient).all() or not np.isfinite(prior_loss + likelihood_loss):
        raise ValueError('Non-finite randomized posterior objective/gradient')
    return prior_loss + likelihood_loss, gradient, prediction, prior_loss, likelihood_loss, sensitivity


def gauss_newton_step(z, residual, sensitivity, damping=0., gram=None):
    """Solve [(1+lambda)I + A.T A] s = -(z+A.T r) in data space."""
    if not np.isfinite(damping) or damping < 0:
        raise ValueError('Gauss-Newton damping must be finite and nonnegative')
    alpha = 1. + damping
    if gram is None:
        gram = sensitivity @ sensitivity.T
    system = gram + alpha * np.eye(len(residual))
    rhs = residual - sensitivity @ z / alpha
    return -z / alpha - sensitivity.T @ cho_solve(cho_factor(system, lower=True), rhs)


def atomic_npz(path, **values):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def optimize_member(prior, center, target, variance, evaluate, checkpoint=None,
                    identity='', max_iterations=50, max_trials=10, max_seconds=13800.,
                    gradient_tolerance=1e-5, initial_damping=0.):
    """Use Jacobian-based damped Gauss-Newton for the full randomized loss."""
    if (max_iterations < 1 or max_trials < 1 or max_seconds <= 0
            or gradient_tolerance <= 0 or initial_damping < 0
            or not np.isfinite(max_seconds + gradient_tolerance + initial_damping)):
        raise ValueError('Optimizer budgets and tolerance must be finite and positive')
    center, target, variance = map(np.asarray, (center, target, variance))
    if (center.shape != prior.mean.shape or target.ndim != 1 or variance.shape != target.shape
            or not np.isfinite(center).all() or not np.isfinite(target).all()
            or not np.isfinite(variance).all() or np.any(variance <= 0)):
        raise ValueError('Require finite prior center and data with positive variances')
    z = np.zeros_like(center)
    history = []
    trials = []
    damping = initial_damping
    state = None
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved['identity']) != identity:
                raise ValueError('Checkpoint belongs to different inputs or settings')
            if 'optimizer' not in saved or str(saved['optimizer']) != OPTIMIZER:
                raise ValueError('Checkpoint belongs to a different optimizer')
            z = saved['z'].copy()
            history = json.loads(str(saved['history']))
            trials = json.loads(str(saved['trial_history']))
            damping = float(saved['damping'])
            prediction, sensitivity = saved['prediction'].copy(), saved['sensitivity'].copy()
            prior_loss = .5 * float(z @ z)
            residual = (prediction - target) / np.sqrt(variance)
            likelihood_loss = .5 * float(residual @ residual)
            state = (prior_loss + likelihood_loss, z + sensitivity.T @ residual,
                     prediction, prior_loss, likelihood_loss, sensitivity)
    started = time.monotonic()
    evaluations = 0

    def evaluate_point(point):
        nonlocal evaluations
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError('Optimizer walltime budget exhausted')
        result = linearize(point, center, target, variance, prior, evaluate)
        evaluations += 1
        return result

    def save():
        if checkpoint is not None:
            atomic_npz(checkpoint, z=z, prediction=state[2], sensitivity=state[5],
                       identity=identity, optimizer=OPTIMIZER, damping=damping,
                       history=json.dumps(history), trial_history=json.dumps(trials))

    def record(step_inf, used_damping=None, gain_ratio=None):
        loss, gradient, prediction, prior_loss, likelihood_loss, _ = state
        row = {'objective': loss, 'prior_loss': prior_loss, 'likelihood_loss': likelihood_loss,
               'gradient_inf': float(np.max(np.abs(gradient))),
               'step_inf': step_inf, 'damping': used_damping, 'gain_ratio': gain_ratio}
        history.append(row)
        save()

    if state is None:
        state = evaluate_point(z)
        record(0.)
    reason = 'iteration_limit'
    message = ''
    try:
        while True:
            loss, gradient, prediction, _, _, sensitivity = state
            if np.max(np.abs(gradient)) <= gradient_tolerance:
                reason = 'gradient_tolerance'
                break
            if len(history) - 1 >= max_iterations:
                break
            residual = (prediction - target) / np.sqrt(variance)
            gram = sensitivity @ sensitivity.T
            for trial in range(max_trials):
                used_damping = damping
                step = gauss_newton_step(z, residual, sensitivity, damping, gram)
                projected = sensitivity @ step
                predicted_reduction = float(-gradient @ step - .5 * (step @ step + projected @ projected))
                if not np.isfinite(predicted_reduction) or predicted_reduction <= 0:
                    reason, message = 'stalled', 'No positive Gauss-Newton predicted reduction'
                    break
                candidate_state = evaluate_point(z + step)
                actual_reduction = loss - candidate_state[0]
                gain_ratio = actual_reduction / predicted_reduction
                accepted = actual_reduction > 0 and gain_ratio >= 1e-4
                trials.append({'iteration': len(history), 'trial': trial + 1,
                               'damping': used_damping, 'accepted': accepted,
                               'predicted_reduction': predicted_reduction,
                               'actual_reduction': actual_reduction, 'gain_ratio': gain_ratio})
                if accepted:
                    if gain_ratio > .75:
                        damping /= 3.
                    elif gain_ratio < .25:
                        damping = max(1e-3, 2. * damping)
                    z = z + step
                    state = candidate_state
                    record(float(np.max(np.abs(step))), used_damping, gain_ratio)
                    break
                damping = max(1e-3, 10. * damping)
                save()
            else:
                reason, message = 'no_acceptable_trial', 'Damping trials exhausted without objective decrease'
                break
            if reason == 'stalled':
                break
    except TimeoutError as error:
        reason, message = 'time_limit', str(error)
    return {'model': center + factor_apply(prior, z), 'z': z, 'prediction': state[2],
            'optimizer': OPTIMIZER, 'next_damping': damping, 'trial_history': trials,
            'accepted_iterations': len(history) - 1,
            'jacobian_evaluations_this_attempt': evaluations,
            'converged': reason == 'gradient_tolerance', 'reason': reason,
            'message': message, 'history': history, 'evaluations_this_attempt': evaluations,
            'seconds_this_attempt': time.monotonic() - started}


LM_OPTIMIZER = 'whitened_parallel_trust_region_lm_v1'
MINIMUM_DAMPING = 1e-6


def step_size(prior, step):
    """Largest raw-state change (ln Rh / anisotropy latent) of a whitened step."""
    return float(np.max(np.abs(factor_apply(prior, step))))


def trust_damping(prior, z, residual, sensitivity, gram, radius, floor):
    """Smallest damping >= floor whose step changes no raw state entry by more than radius."""
    if radius <= 0 or floor <= 0:
        raise ValueError('Trust radius and damping floor must be positive')

    def size(damping):
        return step_size(prior, gauss_newton_step(z, residual, sensitivity, damping, gram))
    if size(floor) <= radius:
        return floor
    low, high = floor, floor * 10.
    while size(high) > radius:
        low, high = high, high * 10.
        if high > 1e14:
            raise ValueError('Trust radius cannot be met by damping')
    while high / low > 1.01:
        middle = np.sqrt(low * high)
        low, high = (middle, high) if size(middle) > radius else (low, middle)
    return high


def candidate_dampings(prior, z, residual, sensitivity, gram, radius, floor, count):
    """Dampings for nested trust radii radius, radius/2, ...; strictly increasing."""
    dampings = []
    for index in range(count):
        damping = trust_damping(prior, z, residual, sensitivity, gram, radius / 2 ** index, floor)
        dampings.append(max(damping, 2. * dampings[-1]) if dampings else damping)
    return dampings


def optimize_member_lm(prior, center, target, variance, evaluate_batch, checkpoint=None,
                       identity='', candidates=4, initial_trust=.3, max_trust=2., min_trust=1e-3,
                       max_iterations=50, max_rounds=4, max_seconds=3300.,
                       stagnation_tolerance=1e-3, stagnation_iterations=2,
                       relative_gradient_tolerance=1e-3, acceptance=1e-4, start=None):
    """Trust-region Levenberg-Marquardt for one randomized posterior, parallel candidates.

    Each round evaluates ``candidates`` damped Gauss-Newton steps at once (forward
    and Jacobian). Their dampings meet nested trust radii on the largest raw-state
    change. The accepted candidate (lowest objective among gain ratio >=
    ``acceptance``) supplies the next linearization; damping follows Nielsen's
    rule and the trust radius adapts to the gain ratio. A round without an
    acceptable candidate shrinks the radius; ``max_rounds`` such rounds in a
    row stop the member as ``nonlinearity_limited``.

    ``start`` (raw model) begins the search away from the prior centre; the
    objective is unchanged (the prior term is still measured from ``center``).
    """
    if (candidates < 1 or max_iterations < 1 or max_rounds < 1 or max_seconds <= 0
            or not 0 < min_trust <= initial_trust <= max_trust or stagnation_iterations < 1
            or stagnation_tolerance <= 0 or relative_gradient_tolerance <= 0 or not 0 < acceptance < .25):
        raise ValueError('Invalid Levenberg-Marquardt settings')
    center, target, variance = map(np.asarray, (center, target, variance))
    if (center.shape != prior.mean.shape or target.ndim != 1 or variance.shape != target.shape
            or not np.isfinite(center).all() or not np.isfinite(target).all()
            or not np.isfinite(variance).all() or np.any(variance <= 0)):
        raise ValueError('Require finite prior center and data with positive variances')
    z = np.zeros_like(center)
    if start is not None:
        start = np.asarray(start, dtype=float)
        if start.shape != center.shape or not np.isfinite(start).all():
            raise ValueError('Start model must be finite and match the prior centre')
        z = factor_solve(prior, start - center)
    history, trials = [], []
    damping, trust, stagnant, initial_gradient = MINIMUM_DAMPING, float(initial_trust), 0, None
    state = None
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved['identity']) != identity:
                raise ValueError('Checkpoint belongs to different inputs or settings')
            if 'optimizer' not in saved or str(saved['optimizer']) != LM_OPTIMIZER:
                raise ValueError('Checkpoint belongs to a different optimizer')
            z = saved['z'].copy()
            history = json.loads(str(saved['history']))
            trials = json.loads(str(saved['trial_history']))
            damping, trust = float(saved['damping']), float(saved['trust'])
            stagnant, initial_gradient = int(saved['stagnant']), float(saved['initial_gradient'])
            state = state_from(z, saved['prediction'], saved['jacobian'], target, variance, prior)
            jacobian = saved['jacobian'].copy()
    started = time.monotonic()
    evaluations = 0

    def evaluate_points(points):
        nonlocal evaluations
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError('Optimizer walltime budget exhausted')
        results = evaluate_batch([center + factor_apply(prior, point) for point in points])
        evaluations += len(points)
        if len(results) != len(points):
            raise ValueError('Batch evaluator returned the wrong number of results')
        states = []
        for point, result in zip(points, results):
            try:
                states.append(None if result is None else
                              (state_from(point, *result, target, variance, prior), np.asarray(result[1])))
            except ValueError:
                states.append(None)
        return states

    def save():
        if checkpoint is not None:
            atomic_npz(checkpoint, z=z, prediction=state[2], jacobian=jacobian, identity=identity,
                       optimizer=LM_OPTIMIZER, damping=damping, trust=trust, stagnant=stagnant,
                       initial_gradient=initial_gradient, history=json.dumps(history),
                       trial_history=json.dumps(trials))

    def record(**extra):
        loss, gradient, _, prior_loss, likelihood_loss, _ = state
        gradient_inf = float(np.max(np.abs(gradient)))
        history.append({'objective': loss, 'prior_loss': prior_loss, 'likelihood_loss': likelihood_loss,
                        'gradient_inf': gradient_inf,
                        'relative_gradient': gradient_inf / initial_gradient, 'trust': trust,
                        'next_damping': damping, **extra})
        save()

    if state is None:
        first = evaluate_points([z])[0]
        if first is None:
            raise RuntimeError('The starting-model evaluation failed')
        state, jacobian = first
        initial_gradient = max(float(np.max(np.abs(state[1]))), 1e-300)
        record(step_inf=0.)
    reason, message = 'iteration_limit', ''
    try:
        while True:
            loss, gradient, prediction, _, _, sensitivity = state
            if np.max(np.abs(gradient)) <= relative_gradient_tolerance * initial_gradient:
                reason = 'relative_gradient'
                break
            if stagnant >= stagnation_iterations:
                reason = 'objective_stagnation'
                break
            if len(history) - 1 >= max_iterations:
                break
            residual = (prediction - target) / np.sqrt(variance)
            gram = sensitivity @ sensitivity.T
            accepted = False
            for round_index in range(max_rounds):
                dampings = candidate_dampings(prior, z, residual, sensitivity, gram, trust,
                                              damping, candidates)
                steps = [gauss_newton_step(z, residual, sensitivity, value, gram) for value in dampings]
                predicted = [float(-gradient @ step - .5 * (step @ step + np.sum((sensitivity @ step) ** 2)))
                             for step in steps]
                sizes = [step_size(prior, step) for step in steps]
                outcomes = evaluate_points([z + step for step in steps])
                rows = []
                for value, step_inf, gain_model, outcome in zip(dampings, sizes, predicted, outcomes):
                    row = {'iteration': len(history), 'round': round_index + 1, 'damping': value,
                           'step_inf': step_inf, 'trust': trust, 'predicted_reduction': gain_model,
                           'failed': outcome is None}
                    if outcome is not None:
                        actual = loss - outcome[0][0]
                        row.update(objective=outcome[0][0], actual_reduction=actual,
                                   gain_ratio=actual / gain_model if gain_model > 0 else None)
                        row['accepted_candidate'] = bool(gain_model > 0 and actual > 0
                                                         and actual / gain_model >= acceptance)
                    else:
                        row['accepted_candidate'] = False
                    rows.append(row)
                trials.extend(rows)
                good = [index for index, row in enumerate(rows) if row['accepted_candidate']]
                if not good:
                    trust = max(min_trust, min(sizes) / 4.)
                    damping = max(dampings) * 2.
                    save()
                    continue
                best = min(good, key=lambda index: rows[index]['objective'])
                ratio, size = rows[best]['gain_ratio'], sizes[best]
                damping = max(MINIMUM_DAMPING, dampings[best] * max(1 / 3, 1 - (2 * ratio - 1) ** 3))
                if ratio > .75:
                    trust = min(max_trust, max(trust, 2. * size))
                elif ratio < .25:
                    trust = max(min_trust, size / 2.)
                decrease = (loss - rows[best]['objective']) / max(abs(loss), 1e-300)
                stagnant = stagnant + 1 if decrease < stagnation_tolerance else 0
                z = z + steps[best]
                state, jacobian = outcomes[best]
                record(step_inf=size, damping=dampings[best], gain_ratio=ratio,
                       relative_decrease=decrease, round=round_index + 1, candidate=best)
                accepted = True
                break
            if not accepted:
                reason = 'nonlinearity_limited'
                message = f'{max_rounds} rounds of {candidates} candidates without objective decrease'
                break
    except TimeoutError as error:
        reason, message = 'time_limit', str(error)
    return {'model': center + factor_apply(prior, z), 'z': z, 'prediction': state[2],
            'optimizer': LM_OPTIMIZER, 'next_damping': damping, 'trust': trust,
            'trial_history': trials, 'accepted_iterations': len(history) - 1,
            'converged': reason in ('relative_gradient', 'objective_stagnation'),
            'reason': reason, 'message': message, 'history': history,
            'evaluations_this_attempt': evaluations,
            'seconds_this_attempt': time.monotonic() - started}
