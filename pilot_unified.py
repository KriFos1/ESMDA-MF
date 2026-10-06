"""Matched unified-domain pilots: window size, ensemble size, drift diagnostics.

Compares the unified moving-window controller (branch 2D-curtain) against the
staged r2 baseline checkpoints at identical indices: zone waves at nx=15
(matched window) and nx=33 (recommended window) with 250 members, nx=33 with
400 members (rank check), plus forced-2D diagnostics at index 80 with
independent posterior forward checks. Drift diagnostics: per-distance ensemble
spread, world-anchored behind-column drift across each zone, and a no-update
drift control per ensemble size.
"""

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import h5py
import numpy as np

from pilot_projection import ROOT, run_wave
from tv_regularization import value_and_gradient

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

R2 = ROOT / 'runs/benchmark3-full-20261005-r2'
COMMON = ['--sim-backend', 'slurm', '--host', 'HPC', '--partition', 'comp', '--account',
          'energytech', '--qos', 'normal', '--nodes', '1', '--ranks', '1', '--cpus', '5',
          '--memory', '16', '--walltime', '00:05:00', '--concurrency', '5',
          '--poll-max-delay', '5', '--binary-dir', '/cluster/home/krfo/adjoint-2d-v1',
          '--solver-tolerance', '1e-6']
CONTROLS = ['--data', str(ROOT / 'runs/benchmark3-data-20261002/data.pkl'),
            '--variance', str(ROOT / 'runs/benchmark3-data-20261002/var.pkl'),
            '--seed', '2026', '--opt-members', '5', '--misfit-threshold', '100',
            '--single-update', '--prior-style', 'blocky', '--prior-projection', 'off',
            '--regularization', 'tv', '--tv-weight-1d', '.1', '--tv-weight-2d', '.05',
            '--tv-max-step', '.25', '--max-substitutions', '8']
CASES = {
    'u15': ['--nx', '15', '--ensemble-size', '250'],
    'u33': ['--nx', '33', '--ensemble-size', '250'],
    'u33e400': ['--nx', '33', '--ensemble-size', '400'],
}
LM_CONTROLS = [flag for flag in CONTROLS if flag not in ('--single-update',)]
LM_CONTROLS += ['--regularization', 'none', '--max-iterations', '3']
ZONES = [27, 80, 160]


def bit_columns(result):
    nx, _, nz = result['dims']
    params = np.asarray(result['posterior_params'], dtype=float)
    if result.get('unified_domain'):
        return params[:, :nx * nz].reshape(len(params), nx, nz)[:, nx // 2, :]
    if result['selected_fidelity'] == '2D':
        return params[:, :nx * nz].reshape(len(params), nx, nz)[:, nx // 2, :]
    return params[:, :nz]


def spread_by_distance(result):
    if not result.get('unified_domain'):
        return None
    nx, _, nz = result['dims']
    rh = np.asarray(result['posterior_params'], dtype=float)[:, :nx * nz] \
        .reshape(-1, nx, nz)
    bit = nx // 2
    return {str(distance): float(rh[:, bit + distance, :].std(axis=0).mean())
            for distance in (-6, 0, 6, 12) if 0 <= bit + distance < nx}


def truth_target(result, truth, x, z, wx):
    nx, _, nz = result['dims']
    index = result['logging_index']
    column = int(np.argmin(abs(x - wx[index])))
    top = result['shift_ft']['z'] / 3.28084
    dz = result['spacing_ft'][2] / 3.28084
    return np.array([np.mean(truth[(z >= top + k * dz) & (z < top + (k + 1) * dz), column])
                     for k in range(nz)])


def assess_checkpoints(directory, truth, x, z, wx):
    paths = sorted(directory.glob('inversion_results_assim_*.pkl'),
                   key=lambda path: int(path.stem.rsplit('_', 1)[1]))
    rows, results = [], []
    for path in paths:
        result = pickle.loads(path.read_bytes())
        results.append(result)
        target = truth_target(result, truth, x, z, wx)
        rh = bit_columns(result)
        error = (rh - target) / np.log(10.)
        kind = 'forward' if result['posterior_forward_verified'] else 'linearized'
        rows.append({'index': result['logging_index'], 'fidelity': result['selected_fidelity'],
                     'two_d_mode': result.get('two_d_mode'),
                     'rmse_log10': float(np.sqrt(np.mean(error.mean(0) ** 2))),
                     'near_truth_fraction': float(np.mean(abs(error) < .3)),
                     'mean_max_jump_log10': float(np.max(abs(np.diff(rh, axis=1)),
                                                         axis=1).mean() / np.log(10.)),
                     'column_tv': float(np.mean([value_and_gradient(
                         column, (result['dims'][2],), (result['spacing_ft'][2],))[0]
                         for column in rh])),
                     'one_d_final_misfit': result['misfit']['1D_final'],
                     'after_final': result['misfit']['after_final'],
                     f'{kind}_posterior_misfit': float(np.mean(result['posterior_misfit'])),
                     'spread': spread_by_distance(result),
                     'substitutions': result['substituted_members'],
                     'two_d_calls': result['two_d_calls'],
                     'point_seconds': result['point_elapsed_seconds']})
    drift = None
    if len(results) >= 2 and results[-1].get('unified_domain'):
        first, last = results[0], results[-1]
        nx = first['dims'][0]
        world = first['shift_ft']['x'] + (nx // 2 + .5) * first['spacing_ft'][0]
        column = int((world - last['shift_ft']['x']) / last['spacing_ft'][0] - .5)
        if 0 <= column < nx:
            cells = nx * first['dims'][2]
            first_col = np.asarray(first['posterior_params'], dtype=float)[:, :cells] \
                .reshape(-1, nx, first['dims'][2])[:, nx // 2, :]
            last_col = np.asarray(last['posterior_params'], dtype=float)[:, :cells] \
                .reshape(-1, nx, last['dims'][2])[:, column, :]
            drift = float(np.abs(last_col.mean(0) - first_col.mean(0)).mean())
    return rows, drift


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--zone-budget', type=float, default=900.)
    parser.add_argument('--forced-budget', type=float, default=900.)
    parser.add_argument('--check-budget', type=float, default=360.)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    for zone in ZONES:
        for case, options in CASES.items():
            name = f'{case}-{zone}'
            directory = output / name
            command = [sys.executable, '-B', '-u', str(ROOT / 'run_DA_sequential.py'),
                       '--output-dir', str(directory), '--start', str(zone),
                       '--stop', str(zone + 3), *COMMON, *CONTROLS, *options]
            commands.append((name, command, directory, output / f'{name}.log'))
    forced = []
    for case, options in CASES.items():
        name = f'forced-{case}-80'
        directory = output / name
        command = [sys.executable, '-B', '-u', str(ROOT / 'run_DA_sequential.py'),
                   '--output-dir', str(directory), '--start', '80', '--stop', '81',
                   '--force-2d', *COMMON, *CONTROLS, *options]
        forced.append((name, command, directory, output / f'{name}.log'))
    report = {'configuration': {key: str(value) for key, value in vars(args).items()},
              'commands': {name: command for name, command, _, _ in commands},
              'forced_commands': {name: command for name, command, _, _ in forced},
              'cases': CASES, 'zones': ZONES, 'baseline': str(R2),
              'pilots': {}, 'baseline_rows': {}, 'forced': {}, 'forward_checks': {},
              'drift': {}, 'drift_control': {}, 'full_relaunch': False}
    # Local 1D simulators are CPU-bound; parallel controller processes thrash
    # BLAS thread pools, so waves run in small chunks with single-threaded BLAS.
    report['zone_status'] = {}
    for start in range(0, len(commands), 3):
        chunk = commands[start:start + 3]
        report['zone_status'].update({name: status for name, status
                                      in run_wave(chunk, args.zone_budget).items()})
    lm = ('forced-u15lm-80',
          [sys.executable, '-B', '-u', str(ROOT / 'run_DA_sequential.py'),
           '--output-dir', str(output / 'forced-u15lm-80'), '--start', '80',
           '--stop', '81', '--force-2d', *COMMON, *LM_CONTROLS, *CASES['u15']],
          output / 'forced-u15lm-80', output / 'forced-u15lm-80.log')
    report['forced_status'] = {}
    for entry in forced + [lm]:
        report['forced_status'][entry[0]] = run_wave([entry], args.forced_budget)[entry[0]]
    for ensemble in (250, 400):
        control = output / f'drift-control-{ensemble}.json'
        subprocess.run([sys.executable, '-B', str(ROOT / 'drift_control.py'),
                        '--output', str(control), '--ensemble-size', str(ensemble)],
                       check=True, capture_output=True)
        history = json.loads(control.read_text())['history']
        report['drift_control'][ensemble] = {'spread_initial': history[0]['spread_ln'],
                                             'spread_final': history[-1]['spread_ln'],
                                             'reservoir_fraction_final':
                                                 history[-1]['reservoir_fraction']}
    with h5py.File(ROOT.parent / 'Dashboard/benchmarks/Benchmark-3/globalmodel.h5', 'r') as handle:
        truth = np.log(handle['formation/Rh'][:, handle['formation/Rh'].shape[1] // 2, :])
        x, z, wx = handle['formation/x'][:, 0], handle['formation/z'][:, 0], \
            handle['wellpath/X'][:, 0]
    checks = []
    base_rows, _ = assess_checkpoints(R2, truth, x, z, wx)
    for name, _, directory, _ in commands + forced + [lm]:
        rows, drift = assess_checkpoints(directory, truth, x, z, wx)
        report['pilots'][name] = rows
        if drift is not None:
            report['drift'][name] = drift
        if not name.startswith('forced'):
            index = rows[0]['index'] if rows else int(name.rsplit('-', 1)[1])
            report['baseline_rows'][name] = [row for row in base_rows
                                             if row['index'] == index]
        two_d = [path for path in directory.glob('inversion_results_assim_*.pkl')
                 if pickle.loads(path.read_bytes())['selected_fidelity'] == '2D']
        if two_d:
            check_dir = output / f'check-{name}'
            command = [sys.executable, '-B', '-u', str(ROOT / 'check_posterior.py'),
                       '--checkpoint', str(sorted(two_d)[-1]),
                       '--output-dir', str(check_dir), '--solve-timeout', '150', *COMMON]
            checks.append((name, command, check_dir, output / f'check-{name}.log'))
    report['check_status'] = {}
    for entry in checks:
        report['check_status'][entry[0]] = run_wave([entry], args.check_budget)[entry[0]]
    for name, _, check_dir, _ in checks:
        if (check_dir / 'report.json').exists():
            report['forward_checks'][name] = json.loads(
                (check_dir / 'report.json').read_text())
    (output / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Assessment: {output / "assessment.json"}')
    for name, rows in report['pilots'].items():
        print(name, 'points', len(rows))
        for row in rows:
            print(json.dumps(row))
    return 0 if all(status['returncode'] == 0
                    for status in list(report['zone_status'].values())
                    + list(report['forced_status'].values())) else 1


if __name__ == '__main__':
    raise SystemExit(main())
