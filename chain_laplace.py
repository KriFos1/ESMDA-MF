"""Laplace 2D stage for a gate-open point of the sequential Matern chain.

The chain exports the point (ensemble, data, window) when the 1D probe fails
(probe-and-revert gate: the carried prior, all of the data energy) or when the
cascade sends it to 2D (the 1D-updated ensemble, the remaining share of the
energy, i.e. Cd / likelihood_weight). The MAP is found on HPC by the parallel
trust-region LM of ``pilot_rml_index45.py`` (``prepare-laplace``) under a prior
built from that ensemble (cellwise Matern, or the ensemble's own sparse
precision, ``sparse_precision``), with unperturbed data and the prior mean as
start. ``sample`` then draws the posterior ensemble from the Laplace
approximation around the MAP and writes the chain checkpoint for that index;
``compare`` does the same for an alternative problem without touching the
chain.

Whitened coordinates m = m_pr + L z, J~ = Cd^-1/2 G L with G = dg/dm at the MAP:
the Laplace covariance is C = (I + J~^T J~)^-1, and z = C (eps + J~^T eta) with
eps ~ N(0, I), eta ~ N(0, I) has covariance C; by Woodbury
z = v - J~^T (I + J~ J~^T)^-1 J~ v, so only data-space (nd x nd) solves occur.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from esmda_core import clamp_log_rh, measurement_misfit
from matern_prior import MaternPrior
from pilot_2d_only_index31 import cap_rh

STD_FLOORS = (.23, .1)        # ln Rh, latent: same floors as the earlier ensemble SpatialPrior
RH_CAP_OHMM = 40.
PRECISION_RADIUS = 2          # cells: neighbourhood of the ensemble sparse-precision prior


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_problem(path, *, index, state, observations, variance, mask, shift, movement, dims,
                   spacing, tools, anisotropy, anisotropy_moments, ranges, probe, well,
                   likelihood_weight=1., prior_type='matern_cellwise'):
    """Save everything the Laplace stage needs for one gate-open logging point.

    ``state`` is the ensemble the 2D prior is built from; the 2D stage
    assimilates the ``likelihood_weight`` share of the data energy (Cd /
    weight) under a ``prior_type`` prior ('matern_cellwise' or
    'ensemble_precision').
    """
    if not 0 < likelihood_weight <= 1:
        raise ValueError('The 2D share of the data energy must lie in (0, 1]')
    if prior_type not in ('matern_cellwise', 'ensemble_precision'):
        raise ValueError(f'Unknown 2D prior type {prior_type}')
    values = {'likelihood_weight': np.float64(likelihood_weight), 'prior_type': np.array(prior_type),'prior_ensemble': np.asarray(state, dtype=float), 'observations': np.asarray(observations),
              'variance': np.asarray(variance), 'mask': np.asarray(mask, dtype=bool),
              'shift': np.array([shift['x'], shift['y'], shift['z']], dtype=float),
              'movement': np.asarray(movement, dtype=int), 'dims': np.asarray(dims, dtype=int),
              'spacing': np.asarray(spacing, dtype=float), 'index': np.int64(index),
              'tools': np.array(json.dumps([list(tool) for tool in tools])),
              'anisotropy': np.array(anisotropy),
              'anisotropy_moments': np.asarray(anisotropy_moments, dtype=float),
              'ranges': np.asarray(ranges, dtype=float), 'probe': np.array(json.dumps(probe))}
    values.update({f'well_{key}': np.asarray(value) for key, value in well.items()})
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **values)


def load_problem(path):
    with np.load(path) as saved:
        problem = {key: saved[key].copy() for key in saved.files}
    problem['index'] = int(problem['index'])
    problem['dims'] = [int(n) for n in problem['dims']]
    problem['spacing'] = [float(h) for h in problem['spacing']]
    problem['tools'] = [tuple(tool) for tool in json.loads(str(problem['tools']))]
    problem['anisotropy'] = str(problem['anisotropy'])
    problem['probe'] = json.loads(str(problem['probe']))
    problem['shift_ft'] = dict(zip(('x', 'y', 'z'), (float(v) for v in problem['shift'])))
    # Problems exported before the cascade carry the whole energy under the cellwise Matern prior.
    problem['likelihood_weight'] = float(problem.get('likelihood_weight', 1.))
    problem['prior_type'] = str(problem.get('prior_type', 'matern_cellwise'))
    return problem


def ensemble_moments(state, cells, floors=STD_FLOORS):
    """Cellwise mean and (floored) standard deviation of each block of an ensemble."""
    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape[0] != 2 * cells or state.shape[1] < 2:
        raise ValueError('Ensemble must have shape (2 * cells, members >= 2)')
    means = [state[:cells].mean(axis=1), state[cells:].mean(axis=1)]
    stds = [np.maximum(state[:cells].std(axis=1, ddof=1), floors[0]),
            np.maximum(state[cells:].std(axis=1, ddof=1), floors[1])]
    return means, stds


def fitted_prior(problem, floors=STD_FLOORS):
    """Matern prior with the carried ensemble's cellwise moments and the chain's ranges."""
    nx, _, nz = problem['dims']
    dx, _, dz = problem['spacing']
    means, stds = ensemble_moments(problem['prior_ensemble'], nx * nz, floors)
    return MaternPrior((nx, nz), (dx, dz), problem['ranges'].tolist(), means, stds)


def laplace_samples(prior, center, z_map, jacobian, variance, count, seed):
    """Draw ``count`` whitened Laplace deviations and the corresponding models.

    ``jacobian`` is dg/dm at the MAP (rows = retained channels, columns = raw
    state). Returns models (n x count), whitened deviations z (n x count) and
    the linearized prediction deviations G L z (nd x count).
    """
    variance = np.asarray(variance, dtype=float)
    jl = prior.color(np.asarray(jacobian, dtype=float).T, transpose=True).T / np.sqrt(variance)[:, None]
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(len(prior.mean), count))
    eta = rng.normal(size=(len(variance), count))
    v = eps + jl.T @ eta
    gram = np.eye(len(variance)) + jl @ jl.T
    z = v - jl.T @ np.linalg.solve(gram, jl @ v)
    deviations = prior.color(z)
    models = np.asarray(center, dtype=float)[:, None] + prior.color(np.asarray(z_map, dtype=float))[:, None] \
        + deviations
    return models, z, np.asarray(jacobian, dtype=float) @ deviations


def map_prior(problem, laplace_run, map_study):
    """The prior the MAP run used, rebuilt from the exported problem and checked against the run's file."""
    if map_study['prior']['type'] == 'ensemble_precision':
        from sparse_precision import EnsemblePrecisionPrior
        saved = EnsemblePrecisionPrior.load(laplace_run / map_study['prior']['file'])
        refit = EnsemblePrecisionPrior.fit(problem['prior_ensemble'], problem['dims'][::2], saved.radius,
                                           saved.floors)
        if not (np.allclose(refit.mean, saved.mean) and np.allclose(refit.std, saved.std)):
            raise ValueError('Ensemble-precision prior differs from the one used for the MAP')
        return saved
    prior = fitted_prior(problem, tuple(map_study['prior']['std_floors']))
    with np.load(laplace_run / 'prior-moments.npz') as saved:
        if not (np.allclose(np.concatenate(saved['means']), prior.mean)
                and np.allclose(np.concatenate(saved['stds']), prior.std)):
            raise ValueError('Fitted prior differs from the one used for the MAP')
    return prior


def selected_member(laplace_run):
    """The multi-start MAP member chosen by the controller (``selected-member.json``), else member 0."""
    record = Path(laplace_run) / 'selected-member.json'
    return int(json.loads(record.read_text())['member']) if record.exists() else 0


def laplace_posterior(chain, problem_path, laplace_run, count, seed):
    """Laplace ensemble and diagnostics for one exported problem and its finished MAP run.

    The Hessian uses the data covariance the MAP assimilated (Cd / likelihood
    weight); misfits are reported against the original Cd.
    """
    from pilot_cascade_laplace_index34 import ensemble_diagnostics, matern_reference
    from sequential_matern_1d import spread_summary
    from transition_mda_study import reference_truth, rh_rmse

    problem = load_problem(problem_path)
    study = json.loads((chain / 'study.json').read_text())
    map_study = json.loads((laplace_run / 'study.json').read_text())
    if map_study['chain']['problem_sha256'] != digest(problem_path):
        raise ValueError('The MAP study was prepared from a different exported problem')
    weight = problem['likelihood_weight']
    if not np.isclose(map_study.get('likelihood_weight', 1.), weight):
        raise ValueError('The MAP study assimilated a different share of the data energy')
    member = laplace_run / f'member-{selected_member(laplace_run):03d}'
    result = json.loads((member / 'result.json').read_text())
    with np.load(member / 'fit.npz') as saved:
        model, z_map, prediction = saved['model'].copy(), saved['z'].copy(), saved['prediction'].copy()
    with np.load(member / 'laplace.npz') as saved:
        jacobian = saved['jacobian'].copy()
        if not np.allclose(saved['prediction'], prediction):
            raise ValueError('Jacobian evaluation does not match the MAP prediction')
    nx, ny, nz = problem['dims']
    dx, dy, dz = problem['spacing']
    cells = nx * nz
    prior = map_prior(problem, laplace_run, map_study)
    if not np.allclose(prior.mean + prior.color(z_map), model, atol=1e-8):
        raise ValueError('MAP model is not the prior mean plus the coloured whitened solution')
    observations, variance = problem['observations'], problem['variance']
    started = time.monotonic()
    models, z, linear = laplace_samples(prior, prior.mean, z_map, jacobian, variance / weight, count, seed)
    models = cap_rh(clamp_log_rh(models), RH_CAP_OHMM)
    linearized = prediction[:, None] + linear
    linear_misfit = np.mean((linearized - observations[:, None]) ** 2 / variance[:, None], axis=0)
    map_misfit = measurement_misfit(prediction, observations, variance)
    manifest = json.loads((Path(study['reference']) / 'run.json').read_text())
    climatological = matern_reference(manifest, problem['anisotropy'], *problem['anisotropy_moments'])
    truth = reference_truth(Path(manifest['benchmark']))
    meta = {'dims': (nx, ny, nz), 'spacing_ft': (dx, dy, dz), 'shift_ft': problem['shift_ft']}
    n = len(prior.mean)
    posterior = {'rh_rmse_log10': rh_rmse(models, meta, truth),
                 'map_rh_rmse_log10': rh_rmse(model[:, None], meta, truth),
                 'map_2d_misfit': map_misfit,
                 'linearized_member_misfit_median': float(np.median(linear_misfit)),
                 'linearized_member_misfit_p90': float(np.percentile(linear_misfit, 90)),
                 'whitened_spread_per_dim': float(np.mean(np.sum(z ** 2, axis=0)) / n),
                 'whitened_spread_rh_per_cell': float(np.mean(np.sum(z[:cells] ** 2, axis=0)) / cells),
                 'map_whitened_offset_per_dim': float(z_map @ z_map / n),
                 'prior_rh_rmse_log10': rh_rmse(problem['prior_ensemble'], meta, truth),
                 'likelihood_weight': weight, 'prior_type': map_study['prior']['type'],
                 **ensemble_diagnostics(models, nx, nz, climatological, problem['anisotropy']),
                 **spread_summary(models, nx, nz)}
    return {'problem': problem, 'map_study': map_study, 'result': result, 'model': model,
            'prediction': prediction, 'models': models, 'linear_misfit': linear_misfit,
            'map_misfit': map_misfit, 'posterior': posterior, 'seconds': time.monotonic() - started}


def map_summary(laplace_run, out):
    result = out['result']
    return {'run': str(laplace_run), 'reason': result.get('reason'),
            'accepted_iterations': result.get('accepted_iterations'),
            'seconds': result.get('seconds_this_attempt'), 'misfit': out['map_misfit']}


def sample(args):
    chain = args.chain.resolve()
    laplace_run = args.map.resolve()
    gates = sorted(chain.glob('gate-*.json'))
    if not gates:
        raise FileNotFoundError('No gate record in the chain directory')
    gate = json.loads(gates[-1].read_text())
    index = gate['logging_index']
    out = laplace_posterior(chain, chain / gate['problem'], laplace_run, args.count, args.seed)
    problem, map_study, result = out['problem'], out['map_study'], out['result']
    model, prediction, models, posterior = out['model'], out['prediction'], out['models'], out['posterior']
    map_misfit, weight = out['map_misfit'], problem['likelihood_weight']
    nx, ny, nz = problem['dims']
    dx, dy, dz = problem['spacing']
    checkpoint = chain / f'inversion_results_assim_{index}.pkl'
    probe = problem['probe']
    reverted = bool(probe.get('reverted', True))
    results = {'prior_params': problem['prior_ensemble'], 'post_mda_params': models.copy(),
               'post_1d_params': problem['prior_ensemble'],
               'posterior_ensemble_params': models.T.copy(), 'posterior_params': [model.copy()],
               'posterior_predictions': [prediction.copy()], 'posterior_misfit': [map_misfit],
               'laplace_linearized_misfit': out['linear_misfit'].tolist(),
               'observations': problem['observations'], 'data_covariance': problem['variance'],
               'observation_mask': problem['mask'],
               'masked_channels': int((~problem['mask']).sum()), 'dims': (nx, ny, nz),
               'spacing_ft': (dx, dy, dz), 'shift_ft': problem['shift_ft'],
               'window_movement': tuple(int(m) for m in problem['movement']),
               'previous_index': gate.get('previous_index'), 'logging_index': index,
               'tools': problem['tools'], 'seed': args.seed,
               'misfit': {'1D_initial': probe['pre_update_mean_prediction_misfit'],
                          '1D_final': probe['post_mean_prediction_misfit'],
                          'before_final': probe['post_mean_prediction_misfit'],
                          'after_final': map_misfit, 'after_final_kind': 'forward'},
               'misfit_threshold': probe['threshold'], 'gate': probe, 'reverted': reverted,
               'cascade': gate.get('cascade'),
               'mda_weight': weight, 'lower_fidelity_weight': 0. if reverted else 1. - weight,
               'optimized_2d': True,
               'selected_fidelity': '2D', 'final_stage': 'laplace', 'two_d_mode': 'map-laplace',
               'localization': 'none', 'prior_style': 'matern', 'two_d_prior': map_study['prior']['type'],
               'anisotropy': problem['anisotropy'],
               'rv_ratio_bounds': (1., float(map_study['ratio_max'])), 'rv_hard_cap_ohmm': RH_CAP_OHMM,
               'rh_cap_ohmm': RH_CAP_OHMM, 'parameter_units': 'natural_log', 'unified_domain': True,
               'posterior_prediction_kind': 'forward', 'posterior_misfit_kind': 'forward',
               'posterior_forward_verified': True,
               'laplace': {'map_run': str(laplace_run), 'map_reason': result.get('reason'),
                           'map_accepted_iterations': result.get('accepted_iterations'),
                           'map_seconds': result.get('seconds_this_attempt'),
                           'prior': map_study['prior'], 'count': args.count, **posterior},
               'point_elapsed_seconds': out['seconds']}
    with checkpoint.open('xb') as stream:
        pickle.dump(results, stream)
    history_path = chain / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({'logging_index': index, 'stage': 'laplace_2d', 'gate': probe, 'cascade': gate.get('cascade'),
                    'map': map_summary(laplace_run, out), 'posterior': posterior, 'count': args.count})
    history_path.write_text(json.dumps(history, indent=1) + '\n')
    (chain / f'laplace-{index:03d}.json').write_text(json.dumps(
        {'logging_index': index, 'map': history[-1]['map'], 'posterior': posterior,
         'count': args.count, 'seed': args.seed}, indent=1) + '\n')
    print(json.dumps(brief(index, out)))
    return 0


def brief(index, out):
    posterior = out['posterior']
    return {'index': index, 'map_misfit': round(out['map_misfit'], 3),
            'map_reason': out['result'].get('reason'),
            'linearized_member_misfit_median': round(posterior['linearized_member_misfit_median'], 3),
            'rh_rmse': round(posterior['rh_rmse_log10'], 4),
            'map_rh_rmse': round(posterior['map_rh_rmse_log10'], 4),
            'prior_rh_rmse': round(posterior['prior_rh_rmse_log10'], 4),
            'std_bit': round(posterior['mean_std_ln_rh_bit_column'], 3),
            'score_rh': round(posterior['matern_2p_over_n_rh_median'], 3)}


def compare(args):
    """Laplace ensemble of an alternative problem (e.g. the reverted cascade point), outside the chain."""
    chain, laplace_run, output = args.chain.resolve(), args.map.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    out = laplace_posterior(chain, args.problem.resolve(), laplace_run, args.count, args.seed)
    index = out['problem']['index']
    np.savez(output / 'samples.npz', models=out['models'], model=out['model'], prediction=out['prediction'],
             linearized_misfit=out['linear_misfit'])
    (output / 'summary.json').write_text(json.dumps(
        {'logging_index': index, 'problem': str(args.problem.resolve()), 'probe': out['problem']['probe'],
         'map': map_summary(laplace_run, out), 'posterior': out['posterior'], 'count': args.count,
         'seed': args.seed}, indent=1) + '\n')
    print(json.dumps(brief(index, out)))
    return 0


def carry(args):
    """Fallback when the 2D stage fails: the gate point keeps its carried (reverted) prior."""
    chain = args.chain.resolve()
    gates = sorted(chain.glob('gate-*.json'))
    if not gates:
        raise FileNotFoundError('No gate record in the chain directory')
    gate = json.loads(gates[-1].read_text())
    index = gate['logging_index']
    checkpoint = chain / f'inversion_results_assim_{index}.pkl'
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    problem = load_problem(chain / gate['problem'])
    nx, ny, nz = problem['dims']
    dx, dy, dz = problem['spacing']
    state, probe = problem['prior_ensemble'], problem['probe']
    results = {'prior_params': state, 'post_mda_params': state.copy(), 'post_1d_params': state,
               'posterior_ensemble_params': state.T.copy(), 'posterior_params': state.T.copy(),
               'posterior_predictions': None, 'posterior_misfit': None,
               'observations': problem['observations'], 'data_covariance': problem['variance'],
               'observation_mask': problem['mask'], 'dims': (nx, ny, nz), 'spacing_ft': (dx, dy, dz),
               'shift_ft': problem['shift_ft'], 'window_movement': tuple(int(m) for m in problem['movement']),
               'previous_index': gate.get('previous_index'), 'logging_index': index,
               'tools': problem['tools'],
               'misfit': {'1D_initial': probe['pre_update_mean_prediction_misfit'],
                          '1D_final': probe['post_mean_prediction_misfit'],
                          'before_final': probe['post_mean_prediction_misfit'],
                          'after_final': probe['pre_update_mean_prediction_misfit'],
                          'after_final_kind': 'forward'},
               'misfit_threshold': probe['threshold'], 'gate': probe, 'reverted': True,
               'optimized_2d': False, 'selected_fidelity': '1D', 'final_stage': 'carried_prior_fallback',
               'fallback_reason': args.reason, 'prior_style': 'matern',
               'anisotropy': problem['anisotropy'], 'rv_hard_cap_ohmm': RH_CAP_OHMM,
               'parameter_units': 'natural_log', 'unified_domain': True}
    with checkpoint.open('xb') as stream:
        pickle.dump(results, stream)
    history_path = chain / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({'logging_index': index, 'stage': 'carried_prior_fallback', 'gate': probe,
                    'reason': args.reason})
    history_path.write_text(json.dumps(history, indent=1) + '\n')
    print(json.dumps({'index': index, 'stage': 'carried_prior_fallback', 'reason': args.reason}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sampler = sub.add_parser('sample', help='Laplace-resample the gate point from a finished MAP run')
    sampler.add_argument('--chain', type=Path, required=True)
    sampler.add_argument('--map', type=Path, required=True)
    sampler.add_argument('--count', type=int, default=250)
    sampler.add_argument('--seed', type=int, default=7031)
    comparison = sub.add_parser('compare', help='Laplace-resample an alternative problem without touching '
                                                'the chain (side-by-side checks)')
    comparison.add_argument('--chain', type=Path, required=True)
    comparison.add_argument('--problem', type=Path, required=True)
    comparison.add_argument('--map', type=Path, required=True)
    comparison.add_argument('--output', type=Path, required=True)
    comparison.add_argument('--count', type=int, default=250)
    comparison.add_argument('--seed', type=int, default=7031)
    fallback = sub.add_parser('carry', help='Write the gate point with its carried prior (2D stage failed)')
    fallback.add_argument('--chain', type=Path, required=True)
    fallback.add_argument('--reason', default='2D stage failed')
    args = parser.parse_args(argv)
    if args.command == 'carry':
        return carry(args)
    if args.count < 2:
        parser.error('Require at least two members')
    return compare(args) if args.command == 'compare' else sample(args)


if __name__ == '__main__':
    raise SystemExit(main())
