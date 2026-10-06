"""Local-window geometry and optimizer bookkeeping for sequential inversion."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


LOG_RH_LIMITS = (-1.0, 4.0)


class MemberEvaluationError(RuntimeError):
    """Initial member evaluation failed after retry; caller should substitute."""


def clamp_log_rh(values):
    """Clamp the ln Rh block (first half of the parameter layout) to LOG_RH_LIMITS."""
    values = np.array(values, dtype=float, copy=True)
    half = values.shape[0] // 2
    values[:half] = np.clip(values[:half], *LOG_RH_LIMITS)
    return values


def project_to_modes(values, background, reservoir, radius):
    """Pull ln Rh values toward their nearest mode, capping the deviation.

    Keeps the blocky family anchored: after every ensemble update each cell
    stays within ``radius`` of the background or reservoir level, so the
    inversion only decides *which* mode each cell belongs to.
    """
    values = np.asarray(values, dtype=float)
    if not background < reservoir or radius <= 0:
        raise ValueError('Require background < reservoir and a positive radius')
    nearest = np.where(values > 0.5 * (background + reservoir), reservoir, background)
    return nearest + np.clip(values - nearest, -radius, radius)


def noise_floor_mask(observations, data_covariance, noise_floor=1e-3):
    """True where a channel's noise standard deviation is below the instrument floor.

    Such channels carry no measurement information yet inflate the reduced
    misfit without bound, because the variance scales with the (near-zero)
    signal rather than the instrument resolution.
    """
    observations = np.asarray(observations, dtype=float).ravel()
    covariance = np.asarray(data_covariance, dtype=float).ravel()
    if observations.shape != covariance.shape:
        raise ValueError('Observation and covariance lengths must match')
    if not np.isfinite(observations).all() or not (covariance > 0).all():
        raise ValueError('Observations must be finite and covariance positive')
    return np.sqrt(covariance) < noise_floor


@dataclass
class LocalWindow:
    dims: tuple = (15, 5, 20)
    spacing_ft: tuple = (5.0, 5.0, 5.0)
    anchor_ft: tuple | None = None
    offsets: tuple = (0, 0)

    def __post_init__(self):
        if len(self.dims) != 3 or any(int(n) != n or n < 1 for n in self.dims):
            raise ValueError("Window dimensions must be three positive integers")
        if len(self.spacing_ft) != 3 or not np.all(np.isfinite(self.spacing_ft)) or min(self.spacing_ft) <= 0:
            raise ValueError("Window cell sizes must be positive and finite")

    def update(self, position_ft):
        position = np.asarray(position_ft, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Expected finite well coordinates (x, y, z)")
        if self.anchor_ft is None:
            self.anchor_ft = tuple(position)
        dx, dy, dz = self.spacing_ft
        new_offsets = tuple(int(np.floor(delta / step + 0.5)) for delta, step in
                            zip(position[[0, 2]] - np.asarray(self.anchor_ft)[[0, 2]], (dx, dz)))
        movement = tuple(a - b for a, b in zip(new_offsets, self.offsets))
        self.offsets = new_offsets
        nx, ny, nz = self.dims
        shift = {
            "x": self.anchor_ft[0] + new_offsets[0] * dx - (nx // 2 + 0.5) * dx,
            "y": position[1] - ny * dy / 2,
            "z": self.anchor_ft[2] + new_offsets[1] * dz - (nz // 2 + 0.5) * dz,
        }
        return shift, movement


def shift_posterior(posterior, dims, movement):
    """Reindex a posterior into the new window, extending its edge cells."""
    nx, _, nz = dims
    values = np.asarray(posterior, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 * nx * nz:
        raise ValueError("Posterior must have shape (members, 2*nx*nz)")
    x = np.clip(np.arange(nx) + movement[0], 0, nx - 1)
    z = np.clip(np.arange(nz) + movement[1], 0, nz - 1)
    fields = values.reshape(-1, 2, nx, nz)
    return fields[:, :, x, :][:, :, :, z].reshape(values.shape)


def shift_posterior_1d(posterior, nz, movement):
    """Reindex 1D columns into the shifted window, extending edge cells."""
    values = np.asarray(posterior, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 * nz:
        raise ValueError("Posterior must have shape (members, 2*nz)")
    z = np.clip(np.arange(nz) + movement[1], 0, nz - 1)
    fields = values.reshape(-1, 2, nz)
    return fields[:, :, z].reshape(values.shape)


def optimizer_jacobian(j_log_rh, j_log_rv, latent, min_ratio=1.0, max_ratio=5.0):
    latent = np.asarray(latent, dtype=float).ravel(order="C")
    if j_log_rh.shape != j_log_rv.shape or j_log_rh.shape[1] != latent.size:
        raise ValueError("Jacobian blocks and latent field do not match")
    dlog_ratio = np.log(max_ratio / min_ratio) * norm.pdf(latent)
    return np.hstack([j_log_rh + j_log_rv, j_log_rv * dlog_ratio[None, :]])


def observation_vectors(data, variances, index, tools, width=8):
    observations, covariance = [], []
    for tool in tools:
        datum = np.asarray(data.iloc[index][tool], dtype=float)
        entry = variances.iloc[index][tool]
        if len(entry) == 2 and entry[0] == "ABS":
            variance = entry[1]
        elif len(entry) == len(data) + 1 and entry[0] == "ABS":
            variance = entry[index + 1]
        else:
            raise ValueError(f"Unsupported variance entry for {tool} at index {index}")
        variance = np.asarray(variance, dtype=float)
        if datum.shape != (width,) or variance.shape != (width,):
            raise ValueError(f"Expected {width} channels for {tool} at index {index}")
        if not np.isfinite(datum).all() or not np.isfinite(variance).all() or not (variance > 0).all():
            raise ValueError(f"Invalid observation or variance for {tool} at index {index}")
        observations.append(datum)
        covariance.append(variance)
    return np.concatenate(observations), np.concatenate(covariance)


def measurement_misfit(prediction, observations, data_covariance):
    """Reduced chi-square data misfit against the original measurement covariance."""
    prediction = np.asarray(prediction, dtype=float).ravel()
    observations = np.asarray(observations, dtype=float).ravel()
    covariance = np.asarray(data_covariance, dtype=float).ravel()
    if prediction.shape != observations.shape or covariance.shape != observations.shape:
        raise ValueError("Prediction, observation and covariance lengths must match")
    if not np.isfinite(prediction).all() or not np.isfinite(observations).all():
        raise ValueError("Prediction and observations must be finite")
    if not (covariance > 0).all():
        raise ValueError("Measurement covariance must be positive")
    return float(np.mean((prediction - observations) ** 2 / covariance))


def optimize_member(initial, observations, mean, covariance, data_covariance,
                    evaluate, update, loss, max_iterations=5, max_trials=2,
                    active=None):
    parameters = np.asarray(initial, dtype=float).copy()
    try:
        prediction, jacobian = evaluate(parameters)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        try:
            prediction, jacobian = evaluate(parameters)
        except (RuntimeError, ValueError, FloatingPointError):
            raise MemberEvaluationError(f'Initial evaluation failed after retry: {exc}') from exc
    current_loss = loss(prediction, observations, parameters, mean, covariance, data_covariance)
    if not np.isfinite(current_loss):
        raise ValueError("Initial optimizer loss is not finite")
    damping = 1.0
    history = [float(current_loss)]
    errors = []
    for _ in range(max_iterations):
        accepted = False
        for _ in range(max_trials):
            direction = update(parameters, prediction, mean, jacobian, observations,
                               data_covariance, covariance, damping)
            if active is not None:
                direction = np.where(active, direction, 0.0)
            trial = clamp_log_rh(parameters + direction)
            try:
                trial_prediction, trial_jacobian = evaluate(trial)
                trial_loss = loss(trial_prediction, observations, trial, mean,
                                  covariance, data_covariance)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                trial_loss = np.inf
            if np.isfinite(trial_loss) and trial_loss < current_loss:
                parameters, prediction, jacobian = trial, trial_prediction, trial_jacobian
                current_loss = trial_loss
                damping = max(damping / 10, 1e-6)
                accepted = True
                break
            damping = min(damping * 5, 1e16)
        history.append(float(current_loss))
        if not accepted:
            break
    return {"parameters": parameters, "prediction": prediction, "loss": float(current_loss),
            "history": history, "trial_errors": errors}


def _initialize_members(initials, observations, mean, covariance, data_covariance,
                        evaluate_multi, loss, substitute=None):
    """Evaluate the member priors in batched calls, substituting double failures."""
    initials = [np.asarray(initial, dtype=float).copy() for initial in initials]
    queue = [{'slot': k, 'initial': initials[k], 'obs': observations[k], 'attempts': 0}
             for k in range(len(initials))]
    members = {}
    while queue:
        results = evaluate_multi([entry['initial'] for entry in queue])
        if len(results) != len(queue):
            raise ValueError('evaluate_multi must return one entry per parameter vector')
        retry = []
        for entry, res in zip(queue, results):
            if isinstance(res, Exception):
                entry['attempts'] += 1
                if entry['attempts'] < 2:
                    retry.append(entry)
                    continue
                if substitute is not None:
                    replacement = substitute(entry['slot'], res)
                    if replacement is not None:
                        initial, obs = replacement
                        retry.append({'slot': entry['slot'],
                                      'initial': np.asarray(initial, dtype=float),
                                      'obs': obs, 'attempts': 0})
                        continue
                raise MemberEvaluationError(f'Initial evaluation failed twice: {res}')
            prediction, jacobian = res
            parameters = entry['initial']
            current = loss(prediction, entry['obs'], parameters, mean, covariance, data_covariance)
            if not np.isfinite(current):
                raise ValueError('Initial optimizer loss is not finite')
            members[entry['slot']] = {'parameters': parameters.copy(), 'prediction': prediction,
                                      'jacobian': jacobian, 'observations': entry['obs'],
                                      'loss': float(current), 'damping': 1.0,
                                      'history': [float(current)], 'trial_errors': [],
                                      'rejects': 0, 'small_steps': 0, 'done': False}
        queue = retry
    return members


def optimize_members(initials, observations, mean, covariance, data_covariance,
                     evaluate_multi, update, loss, max_iterations=5, max_rejects=2,
                     active_mask=None, improve_tol=0.05, substitute=None):
    """Batched Levenberg-Marquardt over several members, one call per step.

    ``evaluate_multi`` maps a list of parameter vectors to a list of
    ``(prediction, jacobian)`` tuples, or an ``Exception`` value per entry for
    fields the simulator could not evaluate. ``substitute(slot, exc)`` may
    replace a member whose initial evaluation failed twice by returning
    ``(initial, observations)``, or give up by returning ``None``. A member
    stops early once an accepted step improves its loss by less than
    ``improve_tol`` relative, or after ``max_rejects`` consecutive rejections.
    """
    members = _initialize_members(initials, observations, mean, covariance, data_covariance,
                                  evaluate_multi, loss, substitute)
    for _ in range(max_iterations):
        stepping = [member for member in members.values() if not member['done']]
        if not stepping:
            break
        trials = []
        for member in stepping:
            direction = update(member['parameters'], member['prediction'], mean,
                               member['jacobian'], member['observations'],
                               data_covariance, covariance, member['damping'])
            if active_mask is not None:
                direction = np.where(active_mask, direction, 0.0)
            trials.append(clamp_log_rh(member['parameters'] + direction))
        results = evaluate_multi(trials)
        if len(results) != len(trials):
            raise ValueError('evaluate_multi must return one entry per parameter vector')
        for member, trial, res in zip(stepping, trials, results):
            if isinstance(res, Exception):
                member['trial_errors'].append(f'{type(res).__name__}: {res}')
                member['rejects'] += 1
                member['damping'] = min(member['damping'] * 5, 1e16)
            else:
                trial_loss = loss(res[0], member['observations'], trial, mean,
                                  covariance, data_covariance)
                if np.isfinite(trial_loss) and trial_loss < member['loss']:
                    improvement = (member['loss'] - trial_loss) / max(member['loss'], 1e-300)
                    member.update(parameters=trial, prediction=res[0], jacobian=res[1],
                                  loss=float(trial_loss))
                    member['damping'] = max(member['damping'] / 10, 1e-6)
                    member['rejects'] = 0
                    if improvement < improve_tol:
                        member['small_steps'] += 1
                        if member['small_steps'] >= 2:
                            member['done'] = True
                    else:
                        member['small_steps'] = 0
                else:
                    member['rejects'] += 1
                    member['damping'] = min(member['damping'] * 5, 1e16)
            if member['rejects'] >= max_rejects:
                member['done'] = True
            member['history'].append(float(member['loss']))
    return [members[slot] for slot in sorted(members)]


def update_members_once(initials, observations, mean, covariance, data_covariance,
                        evaluate_multi, update, loss, active_mask=None, damping=1.0,
                        substitute=None):
    """One batched linearized update per member: a single evaluate_multi call.

    Evaluates the priors once (with retry/substitution on failure), applies one
    damped Gauss-Newton step per member, and reports the posterior prediction
    through the linearization ``f(x) + J @ dx``. The next point's evaluation
    re-linearizes around the moved prior, so no solver runs are spent on
    trial steps. Returns records shaped like ``optimize_members``.
    """
    members = _initialize_members(initials, observations, mean, covariance, data_covariance,
                                  evaluate_multi, loss, substitute)
    records = {}
    for slot, member in members.items():
        direction = update(member['parameters'], member['prediction'], mean,
                           member['jacobian'], member['observations'],
                           data_covariance, covariance, damping)
        if active_mask is not None:
            direction = np.where(active_mask, direction, 0.0)
        trial = clamp_log_rh(member['parameters'] + direction)
        prediction = member['prediction'] + member['jacobian'] @ (trial - member['parameters'])
        updated = loss(prediction, member['observations'], trial, mean,
                       covariance, data_covariance)
        if not np.isfinite(updated):
            updated = member['loss']
            trial = member['parameters']
            prediction = member['prediction']
            member['trial_errors'].append('Non-finite single-update loss; prior kept')
        records[slot] = {'parameters': trial, 'prediction': prediction,
                          'loss': float(updated), 'history': [member['loss'], float(updated)],
                          'trial_errors': member['trial_errors'],
                          'initial_prediction': member['prediction'].copy(),
                          'prediction_kind': 'linearized'}
    return [records[slot] for slot in sorted(records)]
