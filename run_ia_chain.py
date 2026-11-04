"""Sequential IA field inversion with an apparent-dip ensemble and a 2D gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from scipy import sparse

from apparent_dip import latent_dip, physical_dip
from chain_laplace import export_problem
from enif_mda import EnIFProbe
from esmda_core import LocalWindow, align_phases, noise_floor_mask, observation_vectors, phase_channels
from esmda_remote2d import UDAR_COMPONENTS
from field_trajectory import along_well_trajectory, read_trajectory
from matern_prior import MaternPrior
from pilot_cascade_laplace_index34 import cascade_inflation, member_misfit_summary
from sequential_matern_1d import entered_cells, reindex_window, relax_to_prior
import run_DA_sequential as sequential
from write_data import read_las


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT.parent / 'IA/inputs'
DATA = ROOT.parent / 'IA/prepared-data'
DIP_STD = float(latent_dip(10.))
NOISE_FLOOR = 1e-3                       # channels with a smaller standard deviation are masked
# Prepared tables without a recorded rule predate the per-tool rule (IA/prepare_data.py).
LEGACY_NOISE = {'rule': 'channel-std', 'fraction': .005, 'noise_floor': NOISE_FLOOR}


def lwd_log_values(path, count):
    values = np.loadtxt(path, comments='%', ndmin=2)
    if values.shape != (count, 3) or not np.array_equal(values[:, 0], np.arange(1, count + 1)):
        raise ValueError('LWD resistivity rows must match the trajectory logging points')
    if not np.isfinite(values).all() or np.any(values[:, 1:] <= 0):
        raise ValueError('LWD resistivities must be positive and finite')
    return np.log(values[:, 1:])


def lwd_means(path, count):
    return lwd_log_values(path, count).mean(axis=1)


def read_settings(path):
    lines = [line.split('!', 1)[0].strip() for line in Path(path).read_text().splitlines()]
    if len(lines) < 30:
        raise ValueError('IA inversion settings require at least 30 lines')
    settings = {'anisoflag': int(lines[1]), 'datasign': int(lines[3]), 'toolflag': int(lines[4]),
                'cell_ft': float(lines[17]), 'dip_bounds': [float(x) for x in lines[20].split()],
                'depth_reference': int(lines[29])}
    if (settings['anisoflag'], settings['datasign'], settings['toolflag'], settings['depth_reference']) != (1, 3, 0, 0):
        raise ValueError('IA chain requires anisotropic, all-tool geosignals with midpoint depth reference')
    if not np.isfinite(settings['cell_ft']) or settings['cell_ft'] <= 0 or len(settings['dip_bounds']) != 2:
        raise ValueError('Invalid IA grid size or dip bounds')
    physical_dip(0., settings['dip_bounds'])
    return settings


def noise_settings(variance):
    """The noise rule and fraction recorded with the prepared variances."""
    noise = {'rule': variance.attrs.get('noise_rule', LEGACY_NOISE['rule']),
             'fraction': float(variance.attrs.get('noise_fraction', LEGACY_NOISE['fraction'])),
             'noise_floor': NOISE_FLOOR}
    if noise['rule'] not in ('channel-std', 'per-tool-signal', 'cross-tool-signal') or not (
            np.isfinite(noise['fraction']) and noise['fraction'] > 0):
        raise ValueError(f"Unsupported prepared noise model: {noise['rule']} {noise['fraction']}")
    return noise


def noise_sigma(values, noise):
    """Standard deviations of (points, tools, channels) values under ``noise``."""
    if noise['rule'] == 'channel-std':
        return np.broadcast_to(noise['fraction'] * values.std(axis=0), values.shape)
    if noise['rule'] == 'cross-tool-signal':
        return np.broadcast_to(np.maximum(noise['fraction'] * np.abs(values).mean(axis=1, keepdims=True), 1e-12),
                               values.shape)
    return np.maximum(noise['fraction'] * np.abs(values), 1e-12)


def validate_observations(inputs, data, variance, trajectory):
    """Check the prepared tables against the current IA tool settings, LAS files and noise rule."""
    noise = noise_settings(variance)
    settings = np.loadtxt(inputs / 'tool.inp', delimiter=',', skiprows=1, ndmin=2)
    if settings.ndim != 2 or settings.shape[1] != 2 or not np.isfinite(settings).all():
        raise ValueError('Invalid tool settings')
    labels = [(f'{int(freq)}kHz', f'{spacing:g}ft') for freq, spacing in settings]
    if any(freq != int(freq) or min(freq, spacing) <= 0 for freq, spacing in settings) \
            or labels != list(data.columns) or labels != list(variance.columns):
        raise ValueError('Prepared tools do not match the input tool settings')
    sources = [inputs / 'trajectory.DAT', inputs / 'LWD_resistivity.txt', inputs / 'tool.inp',
               inputs / 'inversion_settings.inp']
    logs, readings = [], []
    for column, (freq, spacing) in enumerate(settings):
        path = inputs / 'logdata' / f'UDAR_{int(spacing)}ft_{int(freq)}kHz.las'
        points, values = read_las(path)
        if not np.array_equal(points, trajectory[:, 0]) or not np.array_equal(
                values, np.stack(data.iloc[:, column].to_numpy())):
            raise ValueError(f'Prepared observations do not match {path}')
        logs.append(path)
        readings.append(values)
    expected = noise_sigma(np.stack(readings, axis=1), noise) ** 2          # (points, tools, channels)
    for column, path in enumerate(logs):
        rows = variance.iloc[:, column].tolist()
        if len(rows) != len(expected) or any(row[0] != 'ABS' or not np.allclose(row[1], target, rtol=1e-12, atol=0)
                                             for row, target in zip(rows, expected[:, column])):
            raise ValueError(f"Prepared variances do not match the {noise['rule']} noise rule "
                             f"({noise['fraction']:g}): {path}")
    sources += logs
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=INPUTS)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stop', type=int, default=379)
    parser.add_argument('--ensemble-size', type=int, default=250)
    parser.add_argument('--mda-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument('--gate-threshold', type=float, default=25.)
    parser.add_argument('--memory-ft', type=float, default=50.)
    parser.add_argument('--nx', type=int, default=33)
    parser.add_argument('--nz', type=int, default=80)
    parser.add_argument('--rh-std', type=float, default=None,
                        help='Override the section-wide log-Rh standard deviation')
    parser.add_argument('--lwd-prior', choices=['section-global', 'pointwise'], default=None,
                        help='New runs default to section-global; resumes retain their original mode')
    parser.add_argument('--ratio-mean', type=float, default=1.2)
    parser.add_argument('--ratio-std', type=float, default=.3)
    parser.add_argument('--dip-std', type=float, default=DIP_STD)
    parser.add_argument('--fixed-zero-dip', action='store_true', help='Fix apparent dip at zero and omit it from the state')
    parser.add_argument('--rh-ranges', type=float, nargs=2, default=[50., 50.], metavar=('X_FT', 'Z_FT'),
                        help='Matern practical ranges of ln Rh along the well and vertically')
    parser.add_argument('--latent-ranges', type=float, nargs=2, default=[80., 35.], metavar=('X_FT', 'Z_FT'),
                        help='Matern practical ranges of the anisotropy latent')
    parser.add_argument('--warm-start', action='store_true',
                        help="Start each 2D MAP at the previous point's MAP model, reindexed to the new window "
                             '(the prior is still the carried ensemble)')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if not (np.isfinite(args.rh_ranges + args.latent_ranges).all() and min(args.rh_ranges + args.latent_ranges) > 0):
        parser.error('Prior ranges must be positive and finite')
    if (args.stop < 1 or args.ensemble_size < 3 or not 1 <= args.mda_steps <= 10
            or args.nx < 2 or args.nz < 2 or not np.isfinite([args.gate_threshold, args.memory_ft,
                    args.ratio_mean, args.ratio_std, args.dip_std]).all()
            or min(args.gate_threshold, args.memory_ft, args.ratio_mean, args.ratio_std, args.dip_std) <= 0
            or (args.rh_std is not None and not (np.isfinite(args.rh_std) and args.rh_std > 0))):
        parser.error('Invalid range, ensemble size, window, prior, or gate settings')
    return args


def driver_args(study, stop):
    return ['--inputs', study['inputs'], '--data-dir', study['data_dir'], '--stop', str(stop),
            '--ensemble-size', str(study['ensemble_size']), '--mda-steps', str(study['mda_steps']),
            '--seed', str(study['seed']), '--gate-threshold', str(study['gate']['threshold']),
            '--memory-ft', str(study['memory_ft']), '--nx', str(study['dims'][0]),
            '--nz', str(study['dims'][2]), '--rh-std', str(study['rh_std']),
            '--ratio-mean', str(study['ratio_mean']), '--ratio-std', str(study['ratio_std']),
            '--dip-std', str(study['dip_std']), '--lwd-prior', study.get('lwd_prior_mode', 'pointwise'),
            '--rh-ranges', *(str(v) for v in study.get('ranges', [[50., 50.], [80., 35.]])[0]),
            '--latent-ranges', *(str(v) for v in study.get('ranges', [[50., 50.], [80., 35.]])[1]), '--resume',
            *(['--fixed-zero-dip'] if study.get('fixed_zero_dip', False) else []),
            *(['--warm-start'] if study.get('warm_start', False) else [])]


def previous_map_model(output, previous_index, nx, nz, movement):
    """The previous point's 2D MAP model reindexed to the current window, or None without a 2D stage there."""
    if previous_index is None or not (output / f'laplace-{previous_index:03d}.json').exists():
        return None
    record = output / f'laplace-{previous_index:03d}.json'
    run = Path(json.loads(record.read_text())['map']['run'])
    selected = run / 'selected-member.json'
    member = json.loads(selected.read_text())['member'] if selected.exists() else 0
    with np.load(run / f'member-{member:03d}' / 'fit.npz', allow_pickle=False) as saved:
        model = saved['model'].astype(float)
    cells = nx * nz
    fields, rest = model[:2 * cells], model[2 * cells:]
    return np.concatenate([reindex_window(fields[:, None], nx, nz, movement)[:, 0], rest])


def main(argv=None):
    args = parse_args(argv)
    inputs, data_dir, output = args.inputs.resolve(), args.data_dir.resolve(), args.output_dir.resolve()
    trajectory = read_trajectory(inputs / 'trajectory.DAT')
    settings = read_settings(inputs / 'inversion_settings.inp')
    cell, dip_bounds = settings['cell_ft'], settings['dip_bounds']
    well, plane = along_well_trajectory(trajectory)
    lwd = lwd_log_values(inputs / 'LWD_resistivity.txt', len(trajectory))
    saved_study = json.loads((output / 'study.json').read_text()) if args.resume else None
    original_mode = saved_study.get('lwd_prior_mode', 'pointwise') if saved_study else None
    mode = args.lwd_prior or original_mode or 'section-global'
    if original_mode and mode != original_mode:
        raise ValueError('Cannot change LWD prior mode while resuming; start a new run directory')
    global_mean, global_std = float(lwd.mean()), float(lwd.std())
    means = np.full(len(trajectory), global_mean) if mode == 'section-global' else lwd.mean(axis=1)
    rh_std = args.rh_std if args.rh_std is not None else (global_std if mode == 'section-global' else float(np.log(2.)))
    if not np.isfinite(rh_std) or rh_std <= 0:
        raise ValueError('LWD-derived log-Rh standard deviation must be positive and finite')
    data, variance = (pd.read_pickle(data_dir / name) for name in ('data.pkl', 'var.pkl'))
    if args.stop > len(trajectory) or len(data) != len(trajectory) or len(variance) != len(data):
        raise ValueError('Trajectory, observations, variance, and stop must agree')
    if not np.allclose(np.asarray(data.index), trajectory[:, 1], rtol=0, atol=1e-5):
        raise ValueError('Observation depths do not match the trajectory')
    tools = list(data.columns)
    if list(variance.columns) != tools:
        raise ValueError('Observation and variance tools differ')
    input_hashes = validate_observations(inputs, data, variance, trajectory)
    noise = noise_settings(variance)
    nx, nz, cells = args.nx, args.nz, args.nx * args.nz
    sequential.dims = (nx, 5, nz)
    sequential.dX = sequential.dY = sequential.dZ = cell
    sequential.tools = [str(tool) for tool in tools]
    sequential.anisotropy, sequential.log_rv_cap = 'log', None
    sequential.min_ratio, sequential.max_ratio = 1., 1.
    ranges = [list(args.rh_ranges), list(args.latent_ranges)]
    prior = MaternPrior((nx, nz), (cell, cell), ranges,
                        [means[0], np.log(args.ratio_mean)], [rh_std, args.ratio_std])
    precision = prior.precision() if args.fixed_zero_dip else sparse.block_diag(
        (prior.precision(), sparse.csr_matrix([[1 / args.dip_std ** 2]])), format='csc')
    inflation = cascade_inflation(args.mda_steps, 'linear-inflation', 1.)
    study = {'method': 'ia_field_dip_enif', 'inputs': str(inputs), 'data_dir': str(data_dir),
             'stop': args.stop, 'ensemble_size': args.ensemble_size, 'mda_steps': args.mda_steps,
             'seed': args.seed, 'gate': {'threshold': args.gate_threshold}, 'memory_ft': args.memory_ft,
             'dims': [nx, 5, nz], 'spacing_ft': [cell, cell, cell], 'rh_std': rh_std,
             'ratio_mean': args.ratio_mean, 'ratio_std': args.ratio_std, 'dip_std': args.dip_std,
             'dip_bounds': dip_bounds, 'anisotropy': 'log', 'rh_limits': None, 'rv_cap': None,
             'lwd_prior': ('pooled section-wide mean and standard deviation of ln LWD resistivity'
                           if mode == 'section-global' else
                           'geometric mean of the two LWD resistivities at the current point'),
             'plane': plane, 'tools': [list(tool) for tool in tools], 'sim_settings': settings,
             'noise': noise, 'ranges': ranges, 'warm_start': bool(args.warm_start), 'input_sha256': input_hashes}
    if args.fixed_zero_dip:
        study['fixed_zero_dip'] = True
    if mode == 'section-global':
        study['lwd_prior_mode'] = mode
        study['lwd_prior_log_mean'] = global_mean
        study['lwd_prior_log_std'] = rh_std
        study['lwd_prior_sample_count'] = int(lwd.size)
    previous_index, state, dip, history = None, None, None, []
    if args.resume:
        if saved_study.get('fixed_zero_dip', False) != args.fixed_zero_dip:
            raise ValueError('Cannot change dip mode while resuming; start a new run directory')
        for key in ('method', 'inputs', 'data_dir', 'ensemble_size', 'mda_steps', 'seed', 'gate',
                    'memory_ft', 'dims', 'rh_std', 'ratio_mean', 'ratio_std', 'dip_std', 'input_sha256'):
            if saved_study[key] != study[key]:
                raise ValueError(f'Resume configuration differs in {key}')
        if saved_study.get('noise', LEGACY_NOISE) != noise:
            raise ValueError('Resume configuration differs in noise; start a new run directory')
        if saved_study.get('ranges', [[50., 50.], [80., 35.]]) != ranges:
            raise ValueError('Resume configuration differs in prior ranges; start a new run directory')
        if mode == 'section-global':
            for key in ('lwd_prior_mode', 'lwd_prior_log_mean', 'lwd_prior_log_std', 'lwd_prior_sample_count'):
                if saved_study.get(key) != study[key]:
                    raise ValueError(f'Resume configuration differs in {key}')
        checkpoints = sorted(output.glob('inversion_results_assim_*.pkl'),
                             key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        history_path = output / 'history.json'
        if checkpoints:
            with checkpoints[-1].open('rb') as stream:
                saved = pickle.load(stream)
            previous_index, state = saved['logging_index'], saved['post_mda_params']
            dip = None if args.fixed_zero_dip else saved['dip_latent']
            history = json.loads(history_path.read_text()) if history_path.exists() else []
        elif list(output.glob('gate-*.json')):
            raise ValueError('The pending 2D gate must be completed before the driver resumes')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'study.json').write_text(json.dumps(study, indent=2) + '\n')
    window = LocalWindow(sequential.dims, (cell, cell, cell))
    seeds = np.random.SeedSequence(args.seed).spawn(args.stop)
    previous_cwd = Path.cwd()
    try:
        os.chdir(output)
        sequential.setup_simulators(None, sequential.tools, field_wellpath=well,
                                    sim_settings={key: settings[key] for key in ('toolflag', 'datasign', 'anisoflag')})
        for index in range(args.stop):
            shift, movement = window.update(tuple(well[key][index, 0] * sequential.meter_to_feet
                                                   for key in ('X', 'Y', 'Z')))
            if previous_index is not None and index <= previous_index:
                continue
            started = time.monotonic()
            sequential.current_shift = shift
            prior.blocks[0].mean[:] = means[index]
            prior.mean[:cells] = means[index]
            streams = seeds[index].spawn(4)
            rng = np.random.default_rng(streams[1])
            if state is None:
                state = prior.sample(args.ensemble_size, streams[0])
                if not args.fixed_zero_dip:
                    dip = np.random.default_rng(streams[3]).normal(0., args.dip_std, args.ensemble_size)
                rho, entered = 1., cells
            else:
                step = abs(float(trajectory[index, 2] - trajectory[previous_index, 2]))
                rho = float(np.exp(-step / args.memory_ft))
                state = reindex_window(state, nx, nz, movement)
                state = relax_to_prior(prior, state, rho, np.random.default_rng(streams[2]))
                entered_mask = entered_cells(nx, nz, movement)
                state = prior.condition_entered(state, entered_mask.ravel(), streams[0])
                entered = int(entered_mask.sum())
                if not args.fixed_zero_dip:
                    dip = rho * dip + np.sqrt(1. - rho ** 2) * args.dip_std * np.random.default_rng(streams[3]).normal(size=args.ensemble_size)
            prior_state, prior_dip = state.copy(), None if dip is None else dip.copy()
            observations, covariance = observation_vectors(data, variance, index, tools)
            mask = ~noise_floor_mask(observations, covariance, NOISE_FLOOR)
            if not mask.any():
                raise ValueError(f'Index {index}: every channel is masked')
            observations, covariance = observations[mask], covariance[mask]
            phase = phase_channels(mask, UDAR_COMPONENTS)
            enif = EnIFProbe(precision, observations, covariance, inflation, int(rng.integers(2 ** 31 - 1)))

            def predict(fields, dip_latent):
                columns = fields.reshape(2, nx, nz, -1)[:, nx // 2].reshape(2 * nz, -1)
                values = np.column_stack([sequential.simulate_pred_only(columns[:, member], index, '1D',
                                            apparent_dip=0. if dip_latent is None else
                                            float(physical_dip(dip_latent[member], dip_bounds)))[mask]
                                          for member in range(args.ensemble_size)])
                return align_phases(values, observations, phase)

            predicted = predict(state, dip)
            initial = member_misfit_summary(predicted, observations, covariance)
            for alpha in inflation:
                augmented = state if args.fixed_zero_dip else np.vstack([state, dip])
                perturb = rng.normal(observations[:, None], np.sqrt(alpha * covariance)[:, None], predicted.shape)
                augmented += enif.step(augmented, predicted, perturb)
                if args.fixed_zero_dip:
                    state = augmented
                else:
                    state, dip = augmented[:-1], augmented[-1]
                predicted = predict(state, dip)
            final = member_misfit_summary(predicted, observations, covariance)
            gate_open = final['member_1d_misfit_median'] > args.gate_threshold
            probe = {'threshold': args.gate_threshold, 'open': bool(gate_open), 'reverted': bool(gate_open),
                     'pre_update_mean_prediction_misfit': initial['mean_prediction_1d_misfit'],
                     'post_mean_prediction_misfit': final['mean_prediction_1d_misfit'],
                     'post_member_median_misfit': final['member_1d_misfit_median']}
            record = {'logging_index': index, 'previous_index': previous_index, 'shift_ft': shift,
                      'window_movement': list(movement), 'masked_channels': int(np.count_nonzero(~mask)),
                      'entered_cells': entered, 'relaxation_rho': rho,
                      'lwd_prior_mean_ohmm': float(np.exp(means[index])), 'gate': probe,
                      'prior': initial, 'posterior': final, 'seconds': time.monotonic() - started}
            history.append(record)
            (output / 'history.json').write_text(json.dumps(history, indent=1) + '\n')
            if gate_open:
                problem = f'laplace-problem-{index:03d}.npz'
                warm = previous_map_model(output, previous_index, nx, nz, movement) if args.warm_start else None
                export_problem(output / problem, index=index, state=prior_state, observations=observations,
                               variance=covariance, mask=mask, shift=shift, movement=movement,
                               dims=sequential.dims, spacing=(cell, cell, cell), tools=tools, anisotropy='log',
                               anisotropy_moments=(np.log(args.ratio_mean), args.ratio_std), ranges=ranges,
                               probe=probe, well=well, prior_type='ensemble_precision', dip_latent=prior_dip,
                               dip_bounds=dip_bounds, fixed_zero_dip=args.fixed_zero_dip, warm_start=warm)
                (output / f'gate-{index:03d}.json').write_text(json.dumps(
                    {'logging_index': index, 'previous_index': previous_index, 'probe': probe, 'problem': problem,
                     'shift_ft': shift, 'window_movement': list(movement), 'relaxation_rho': rho,
                     'entered_cells': entered}, indent=1) + '\n')
                print(f'Index {index}: 2D gate open; exported {problem}', flush=True)
                break
            member_misfit = np.mean((predicted - observations[:, None]) ** 2 / covariance[:, None], axis=0)
            result = {'prior_params': prior_state, 'post_mda_params': state.copy(),
                      **({'dip_latent': dip.copy(), 'prior_dip_latent': prior_dip} if dip is not None else {}),
                      'posterior_ensemble_params': state.T.copy(),
                      'posterior_params': state.T.copy(), 'posterior_predictions': predicted.T.copy(),
                      'posterior_misfit': member_misfit.tolist(), 'observations': observations,
                      'data_covariance': covariance, 'observation_mask': mask, 'dims': sequential.dims,
                      'spacing_ft': (cell, cell, cell), 'shift_ft': shift, 'window_movement': movement,
                      'previous_index': previous_index, 'logging_index': index, 'tools': tools,
                      'selected_fidelity': '1D', 'optimized_2d': False, 'anisotropy': 'log',
                      'rv_ratio_bounds': (1., 1.), 'rv_hard_cap_ohmm': None, 'rh_cap_ohmm': None,
                      'gate': probe, 'misfit_threshold': args.gate_threshold,
                      'misfit': {'before_final': initial['mean_prediction_1d_misfit'],
                                 'after_final': final['mean_prediction_1d_misfit']},
                      'posterior_prediction_kind': 'forward'}
            with (output / f'inversion_results_assim_{index}.pkl').open('xb') as stream:
                pickle.dump(result, stream)
            previous_index = index
            masked = f"; {record['masked_channels']} channels below the noise floor" if record['masked_channels'] else ''
            angle = 0. if dip is None else np.mean(physical_dip(dip, dip_bounds))
            print(f"Index {index}: 1D median misfit {final['member_1d_misfit_median']:.4g}; "
                  f"mean dip {angle:.2f}°{masked}", flush=True)
    finally:
        os.chdir(previous_cwd)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
