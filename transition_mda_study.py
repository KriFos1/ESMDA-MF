"""Matched, pointwise 0D/1D/2D ES-MDA study on Benchmark-3 indices 25–40.

Each point starts at the incoming, saved 2D ensemble (prior_params) from the
reference run. The 16 points are deliberately independent: no candidate's
posterior is fed into the next point, so threshold comparisons share a prior.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import pickle

import h5py
import numpy as np
import pandas as pd

import run_DA_sequential as sequential
from esmda_core import LOG_RH_LIMITS, LocalWindow, measurement_misfit, observation_vectors
from esmda_remote2d import SimBackendConfig, make_remote2d


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / 'runs/benchmark3-full-two-step-esmda-b025-t75-20261015-r2'
INDICES = range(25, 41)
THRESHOLDS = (25., 50., 75., 100., 150., 250.)
MEMBERS = 25
STEPS = 10


class InsufficientMembers(RuntimeError):
    pass


def inflation_schedule():
    """Linearly decreasing observation-covariance inflation; sum(1/alpha)=1."""
    taper = np.arange(15., 5., -1.)
    alpha = taper * np.reciprocal(taper).sum()
    assert len(alpha) == STEPS and np.isclose(np.reciprocal(alpha).sum(), 1.)
    return alpha


def overlap_weights(centers, edges, width):
    lo, hi = centers - width / 2, centers + width / 2
    weights = np.maximum(0., np.minimum(edges[1:, None], hi) -
                         np.maximum(edges[:-1, None], lo)) / width
    for index in np.flatnonzero(weights.sum(axis=1) == 0):
        middle = (edges[index] + edges[index + 1]) / 2
        weights[index, np.argmin(np.abs(centers - middle))] = 1.
    return weights / weights.sum(axis=1, keepdims=True)


def reference_truth(benchmark):
    with h5py.File(benchmark / 'globalmodel.h5', 'r') as handle:
        x = handle['formation/x'][:, 0] / .3048
        z = handle['formation/z'][:, 0] / .3048
        mid = handle['formation/Rh'].shape[1] // 2
        rh = handle['formation/Rh'][:, mid, :]
        wellpath = {key: value[:] for key, value in handle['wellpath'].items()}
    return x, z, rh, wellpath


def rh_rmse(state, saved, truth):
    x, z, rh, _ = truth
    nx, _, nz = saved['dims']
    dx, _, dz = saved['spacing_ft']
    shift = saved['shift_ft']
    wx = overlap_weights(x, shift['x'] + np.arange(nx + 1) * dx,
                         float(np.median(np.diff(x))))
    wz = overlap_weights(z, shift['z'] + np.arange(nz + 1) * dz,
                         float(np.median(np.diff(z))))
    target = wx @ rh.T @ wz.T
    predicted = np.asarray(state)[:nx * nz].reshape(nx, nz, -1).mean(axis=2)
    return float(np.sqrt(np.mean(((predicted - np.log(target)) / np.log(10.)) ** 2)))


def checkpoint(reference, index):
    path = reference / f'inversion_results_assim_{index}.pkl'
    with path.open('rb') as stream:
        saved = pickle.load(stream)
    if saved['logging_index'] != index or tuple(saved['dims']) != (33, 5, 80):
        raise ValueError(f'Incompatible reference checkpoint: {path}')
    return saved


def score_baseline(reference, truth):
    scores = []
    for index in INDICES:
        saved = checkpoint(reference, index)
        scores.append(rh_rmse(saved['post_mda_params'], saved, truth))
    return float(np.mean(scores))


def score_study(output, threshold, count):
    scores = []
    for index in INDICES:
        path = output / f'point-{index}.json'
        point = json.loads(path.read_text())
        branch = point['branches'][str(count)] if point['gate_misfit'] > threshold else point['branches']['0']
        if branch['status'] != 'ok' or branch['survivors'] < 2:
            raise ValueError(f'Index {index} cannot be scored: {branch}')
        scores.append(branch['rmse_log10_rh'])
    return float(np.mean(scores))


def updater_for(state, cells, threshold):
    from pipt.update_schemes.update_methods_ns.approx_update import approx_update

    ne = state.shape[1]
    updater = approx_update()
    updater.proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
    updater.trunc_energy, updater.lam = .95, 0
    updater.list_states = ['rh', 'rv']
    updater.state_scaling = np.ones(state.shape[0])
    updater.prior_info = {name: {'active': state.shape[0] // 2} for name in updater.list_states}
    if threshold is not None:
        sequential._configure_auto_adaloc(updater, threshold, cells, ne)
    else:
        updater.keys_da = {}
    return updater


def update(state, predicted, observed, covariance, alpha, rng, cells, threshold):
    updater = updater_for(state, cells, threshold)
    inflated = covariance * alpha
    updater.scale_data = np.sqrt(inflated)
    perturbed = rng.normal(observed[:, None], np.sqrt(inflated)[:, None], predicted.shape)
    updater.update(enX=state, enY=predicted, enE=perturbed)
    result = state + updater.step
    np.clip(result[:state.shape[0] // 2], *LOG_RH_LIMITS,
            out=result[:state.shape[0] // 2])
    return result


def main(argv=None):
    config_defaults = SimBackendConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    config_defaults.add_cli(parser)
    parser.set_defaults(concurrency=25)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--start', type=int, default=25)
    parser.add_argument('--stop', type=int, default=41)
    parser.add_argument('--baseline-score', action='store_true')
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--score-threshold', type=float, default=75.)
    parser.add_argument('--score-2d-steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(argv)
    if not 25 <= args.start < args.stop <= 41 or not 1 <= args.score_2d_steps <= 5:
        parser.error('Require 25 <= start < stop <= 41 and 1 <= score-2d-steps <= 5')
    reference = args.reference.resolve()
    manifest = json.loads((reference / 'run.json').read_text())
    benchmark = Path(manifest['benchmark'])
    truth = reference_truth(benchmark)
    if args.baseline_score:
        print(f'{score_baseline(reference, truth):.12g}')
        return
    if args.output_dir is None:
        parser.error('--output-dir is required for --score or inversion')
    output = args.output_dir.resolve()
    if args.score:
        print(f'{score_study(output, args.score_threshold, args.score_2d_steps):.12g}')
        return

    output.mkdir(parents=True, exist_ok=True)
    config = config_defaults.from_args(args)
    if config.backend != 'slurm':
        parser.error('2D experiments require --sim-backend slurm')
    if config.concurrency < 1 or config.concurrency > MEMBERS:
        parser.error('Require 1 <= concurrency <= 25')
    data = pd.read_pickle(ROOT / manifest['data'])
    variances = pd.read_pickle(ROOT / manifest['variance'])
    tools = list(data.columns)
    nx, ny, nz = manifest['dims']
    dx, dy, dz = manifest['spacing_ft']
    cells, bit = nx * nz, nx // 2
    sequential.dims = nx, ny, nz
    sequential.dX, sequential.dY, sequential.dZ = dx, dy, dz
    sequential.tools = [str(tool) for tool in tools]
    sequential.max_ratio = manifest['prior_ratio_max']
    sequential.log_rv_cap = float(np.log(manifest['prior_rv_cap_ohmm']))
    sequential.wellpath = truth[3]
    window = LocalWindow((nx, ny, nz), (dx, dy, dz))
    for point in range(args.start):
        position = tuple(truth[3][key][point, 0] * sequential.meter_to_feet
                         for key in ('X', 'Y', 'Z'))
        window.update(position)
    runtime = {'reference': str(reference), 'indices': [args.start, args.stop - 1],
               'alpha': inflation_schedule().tolist(), 'thresholds': THRESHOLDS,
               'schedule': '3 0D, at least 2 1D, remaining 1D or 1–5 2D',
               'gate_after_step': 5, 'localization': {'type': 'sigm',
                                                    'correlation_threshold': .3},
               'selected_members': MEMBERS, 'pointwise_matched_prior': True,
               'simulator': config.__dict__, 'seed': args.seed}
    (output / 'study.json').write_text(json.dumps(runtime, indent=2, default=str))
    old_cwd = Path.cwd()
    try:
        os.chdir(output)
        sequential.setup_simulators(benchmark, [str(tool) for tool in tools])
        sequential.remote_2d = make_remote2d(
            config, (nx, ny, nz), dx, dy, dz, [ast.literal_eval(str(t)) for t in tools])
        sequential.remote_2d.jobs_dir = str(output / 'jobs')
        for index in range(args.start, args.stop):
            position = tuple(truth[3][key][index, 0] * sequential.meter_to_feet
                             for key in ('X', 'Y', 'Z'))
            sequential.current_shift, _ = window.update(position)
            if (output / f'point-{index}.json').exists():
                print(f'Index {index}: complete; skipping', flush=True)
                continue
            saved = checkpoint(reference, index)
            if not all(np.isclose(saved['shift_ft'][key], sequential.current_shift[key], atol=1e-6)
                       for key in ('x', 'y', 'z')):
                raise ValueError(f'Index {index}: reference window shifted unexpectedly')
            sequential._set_logging_point_state(index)
            observed_full, covariance_full = observation_vectors(data, variances, index, tools)
            mask = np.asarray(saved['observation_mask'], dtype=bool)
            if not np.array_equal(observed_full[mask], saved['observations']):
                raise ValueError(f'Index {index}: observation mask or data mismatch')
            observed, covariance = observed_full[mask], covariance_full[mask]
            rng = np.random.default_rng(args.seed + index)
            state = np.asarray(saved['prior_params'], dtype=float).copy()
            if state.shape != (2 * cells, manifest['ensemble_size']):
                raise ValueError(f'Index {index}: bad reference prior shape')

            def predict_1d(fields):
                columns = fields.reshape(2, nx, nz, -1)[:, bit].reshape(2 * nz, -1)
                return np.column_stack([
                    sequential.simulate_pred_only(columns[:, j], index, '1D')[mask]
                    for j in range(columns.shape[1])])

            def predict_2d(fields):
                models = [sequential._build_param_dict(fields[:, j])[0]
                          for j in range(fields.shape[1])]
                results = sequential.remote_2d.run_pred_multi(
                    models, index, sequential.current_shift, sequential.wellpath)
                values = np.asarray(results, dtype=float)[:, mask].T
                return values

            one_d = {}
            inflation = inflation_schedule()
            for step in range(STEPS):
                if step < 3:
                    initial = np.vstack([state[:cells].mean(axis=0), state[cells:].mean(axis=0)])
                    prediction = np.column_stack([
                        sequential.simulate_pred_only(initial[:, j], index, '0D')[mask]
                        for j in range(initial.shape[1])])
                    changed = update(initial, prediction, observed, covariance,
                                     inflation[step], rng, cells, None)
                    state[:cells] += (changed[0] - initial[0])[None, :]
                    state[cells:] += (changed[1] - initial[1])[None, :]
                    np.clip(state[:cells], *LOG_RH_LIMITS, out=state[:cells])
                else:
                    prediction = predict_1d(state)
                    state = update(state, prediction, observed, covariance,
                                   inflation[step], rng, cells, .3)
                if step >= 4:
                    one_d[step + 1] = state.copy()
            gate_prediction = predict_1d(one_d[5])
            gate = float(measurement_misfit(gate_prediction.mean(axis=1), observed, covariance))
            final_prediction = predict_1d(one_d[10])
            branches = {'0': {'status': 'ok', 'survivors': state.shape[1],
                              'rmse_log10_rh': rh_rmse(one_d[10], saved, truth),
                              'data_misfit': float(measurement_misfit(
                                  final_prediction.mean(axis=1), observed, covariance)),
                              '2d_calls': 0}}
            for count in range(1, 6):
                switch = STEPS - count
                starting = one_d[switch]
                ranked = predict_1d(starting)
                ranked_misfit = np.mean((ranked - observed[:, None]) ** 2 /
                                        covariance[:, None], axis=0)
                chosen = np.argsort(ranked_misfit, kind='stable')[:MEMBERS]
                subset = starting[:, chosen].copy()
                selected = chosen.copy()
                branch_rng = np.random.default_rng(args.seed + index * 100 + count)
                failures = []
                calls = 0
                try:
                    for step in range(switch, STEPS + 1):
                        predicted = predict_2d(subset)
                        calls += subset.shape[1]
                        valid = np.isfinite(predicted).all(axis=0)
                        failures.extend([{'member': int(j), 'step': step}
                                         for j in selected[~valid]])
                        subset, selected, predicted = subset[:, valid], selected[valid], predicted[:, valid]
                        if subset.shape[1] < 2:
                            raise InsufficientMembers('Fewer than two 2D members survived')
                        if step < STEPS:
                            subset = update(subset, predicted, observed, covariance,
                                            inflation[step], branch_rng, cells, .3)
                    branch = {'status': 'ok', 'survivors': int(len(selected)),
                              'rmse_log10_rh': rh_rmse(subset, saved, truth),
                              'data_misfit': float(measurement_misfit(
                                  predicted.mean(axis=1), observed, covariance))}
                except InsufficientMembers as exc:
                    branch = {'status': 'failed', 'error': str(exc),
                              'survivors': int(len(selected))}
                branches[str(count)] = {**branch, 'selected_1d_members': chosen.tolist(),
                                        'surviving_members': selected.tolist(),
                                        'failed_evaluations': failures, '2d_calls': calls,
                                        'switch_after_step': switch}
                print(f'Index {index}, {count} 2D steps: {branch}', flush=True)
            point = {'index': index, 'gate_misfit': gate,
                     'gate_open': {str(t): gate > t for t in THRESHOLDS},
                     'branches': branches}
            (output / f'point-{index}.json').write_text(json.dumps(point, indent=2))
            print(f'Index {index} complete; gate={gate:.4g}', flush=True)
    finally:
        os.chdir(old_cwd)


if __name__ == '__main__':
    main()
