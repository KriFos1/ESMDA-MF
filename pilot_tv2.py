"""Stage-specific TV pilots: held-out unforced zones plus matched forced-2D controls.

The forced cases share one seed and one 1D TV weight, so their 1D assimilation
prefixes are identical and only the 2D TV controls differ.
"""

import argparse
import json
from pathlib import Path
import pickle
import sys

import h5py
import numpy as np

from pilot_projection import ROOT, recovery_metrics, run_wave
from tv_regularization import value_and_gradient


TIMING_KEYS = ('total_seconds', 'submitted_seconds', 'wait_seconds',
               'collect_seconds', 'polls', 'state')


def timing_rows(directory):
    path = directory / 'jobs' / 'timings.jsonl'
    if not path.exists():
        return []
    rows = [{key: record.get(key) for key in TIMING_KEYS}
            for record in map(json.loads, path.read_text().splitlines())]
    for row, status_dir in zip(rows, sorted((directory / 'jobs').glob('*/status.json'))):
        evaluations = json.loads(status_dir.read_text()).get('evaluations', [])
        nodes = sorted({entry.get('node', '?').split('.')[0] for entry in evaluations})
        solve = [entry.get('seconds') for entry in evaluations if entry.get('seconds')]
        row['nodes'] = nodes
        row['member_solve_seconds'] = solve
    return rows


def assess_run(directory, truth, x, z, wx):
    paths = sorted(directory.glob('inversion_results_assim_*.pkl'),
                   key=lambda path: int(path.stem.rsplit('_', 1)[1]))
    rows, check_path = [], None
    for path in paths:
        result = pickle.loads(path.read_bytes())
        row = recovery_metrics(result, truth, x, z, wx)
        nx, _, nz = result['dims']
        params = np.asarray(result['posterior_params'])
        rh = params[:, :nx*nz].reshape(-1, nx, nz)[:, nx//2, :] \
            if result['selected_fidelity'] == '2D' else params[:, :nz]
        row.update(column_tv=float(np.mean([value_and_gradient(column, (nz,),
                                                               (result['spacing_ft'][2],))[0]
                                            for column in rh])),
                   ensemble_spread_ln=float(rh.std(axis=0).mean()),
                   tv_dose=result.get('tv_strength_applied'),
                   tv_weights=result.get('tv_weights_effective'),
                   posterior_tv=float(np.mean(result['posterior_tv']))
                   if 'posterior_tv' in result else None,
                   two_d_calls=result['two_d_calls'], substitutions=result['substituted_members'])
        rows.append(row)
        if result['selected_fidelity'] == '2D':
            check_path = path
    return rows, check_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pilot-budget', type=float, default=480.)
    parser.add_argument('--forced-budget', type=float, default=720.)
    parser.add_argument('--check-budget', type=float, default=360.)
    parser.add_argument('--walltime', default='00:02:00',
                        help='Pilot array-job walltime; node-dependent solve times vary ~2x')
    parser.add_argument('--skip-unforced', action='store_true',
                        help='Run only the forced-2D wave (for extra decision cells)')
    parser.add_argument('--forced-spec', action='append', default=None, metavar='NAME=1D,2D,STEP',
                        help='Override forced cases, e.g. u005-tight=.05,.05,.25')
    parser.add_argument('--stage-1d', default='.2',
                        help='1D TV weight of the unforced stage case')
    parser.add_argument('--skip-forced', action='store_true',
                        help='Run only the unforced zone waves')
    args = parser.parse_args(argv)

    def forced_options(spec):
        name, weights = spec.split('=')
        weight_1d, weight_2d, step = weights.split(',')
        return name, ['--regularization', 'tv', '--tv-weight-1d', weight_1d,
                      '--tv-weight-2d', weight_2d, '--tv-max-step', step]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = ['--sim-backend', 'slurm', '--host', 'HPC', '--partition', 'comp', '--account', 'energytech',
              '--qos', 'normal', '--nodes', '1', '--ranks', '1', '--cpus', '5', '--memory', '16',
              '--walltime', args.walltime, '--concurrency', '5', '--poll-max-delay', '5',
              '--binary-dir', '/cluster/home/krfo/adjoint-2d-v1', '--solver-tolerance', '1e-6']
    controls = ['--data', str(ROOT / 'runs/benchmark3-data-20261002/data.pkl'),
                '--variance', str(ROOT / 'runs/benchmark3-data-20261002/var.pkl'),
                '--ensemble-size', '250', '--opt-members', '5', '--seed', '2026',
                '--misfit-threshold', '100', '--single-update', '--prior-style', 'blocky',
                '--prior-projection', 'off']
    cases = {'baseline': ['--regularization', 'none'],
             'stage': ['--regularization', 'tv', '--tv-weight-1d', args.stage_1d,
                       '--tv-weight-2d', '.05']}
    commands = []
    if not args.skip_unforced:
        for zone in [27, 80, 160]:
            for case, options in cases.items():
                name = f'{case}-{zone}'
                directory = output / name
                command = [sys.executable, '-B', '-u', str(ROOT / 'run_DA_sequential.py'),
                           '--output-dir', str(directory), '--start', str(zone), '--stop', str(zone + 3),
                           *common, *controls, *options]
                commands.append((name, command, directory, output / f'{name}.log'))
    forced_cases = dict(forced_options(spec) for spec in args.forced_spec) if args.forced_spec else {
        'forced-2d0': ['--regularization', 'tv', '--tv-weight-1d', '.2', '--tv-weight-2d', '0'],
        'forced-2d005': ['--regularization', 'tv', '--tv-weight-1d', '.2', '--tv-weight-2d', '.05'],
        'forced-2d005-tight': ['--regularization', 'tv', '--tv-weight-1d', '.2',
                               '--tv-weight-2d', '.05', '--tv-max-step', '.25'],
        'forced-2d010-tight': ['--regularization', 'tv', '--tv-weight-1d', '.2',
                               '--tv-weight-2d', '.1', '--tv-max-step', '.25'],
    }
    forced = [] if args.skip_forced else []
    if not args.skip_forced:
        for case, options in forced_cases.items():
            directory = output / case
            command = [sys.executable, '-B', '-u', str(ROOT / 'run_DA_sequential.py'),
                       '--output-dir', str(directory), '--start', '80', '--stop', '81', '--force-2d',
                       *common, *controls, *options]
            forced.append((case, command, directory, output / f'{case}.log'))
    report = {'configuration': vars(args) | {'output_dir': str(output)},
              'commands': {name: command for name, command, _, _ in commands},
              'forced_commands': {name: command for name, command, _, _ in forced},
              'pilots': {}, 'forward_checks': {}, 'timings': {}, 'full_relaunch': False,
              'poll_cap_seconds': 5., 'forced_scheduling': 'sequential'}
    report['pilot_status'] = run_wave(commands, args.pilot_budget)
    # One forced run at a time: concurrent batches oversubscribe the node and
    # push per-member solves past the pilot walltime (observed TIMED_OUT wave).
    report['forced_status'] = {}
    for entry in forced:
        report['forced_status'][entry[0]] = run_wave([entry], args.forced_budget)[entry[0]]
    with h5py.File(ROOT.parent / 'Dashboard/benchmarks/Benchmark-3/globalmodel.h5', 'r') as handle:
        truth = np.log(handle['formation/Rh'][:, handle['formation/Rh'].shape[1] // 2, :])
        x, z = handle['formation/x'][:, 0], handle['formation/z'][:, 0]
        wx = handle['wellpath/X'][:, 0]
    checks = []
    for name, _, directory, _ in commands + forced:
        rows, check_path = assess_run(directory, truth, x, z, wx)
        report['pilots'][name] = rows
        report['timings'][name] = timing_rows(directory)
        if check_path is not None:
            check_dir = output / f'check-{name}'
            command = [sys.executable, '-B', '-u', str(ROOT / 'check_posterior.py'),
                       '--checkpoint', str(check_path), '--output-dir', str(check_dir),
                       '--solve-timeout', '45', *common]
            checks.append((name, command, check_dir, output / f'check-{name}.log'))
    report['check_commands'] = {name: command for name, command, _, _ in checks}
    report['check_status'] = {}
    for entry in checks:
        report['check_status'][entry[0]] = run_wave([entry], args.check_budget)[entry[0]]
    for name, _, check_dir, _ in checks:
        if (check_dir / 'report.json').exists():
            report['forward_checks'][name] = json.loads((check_dir / 'report.json').read_text())
    (output / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Assessment: {output / "assessment.json"}')
    for name, rows in report['pilots'].items():
        print(name, len(rows), 'points')
        for row in rows:
            print(json.dumps(row))
    return 0 if all(status['returncode'] == 0
                    for status in list(report['pilot_status'].values())
                    + list(report['forced_status'].values())) else 1


if __name__ == '__main__':
    raise SystemExit(main())
