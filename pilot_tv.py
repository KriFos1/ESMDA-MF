"""Bounded TV strength comparison with independent posterior forward checks."""

import argparse
import json
from pathlib import Path
import pickle
import sys

import h5py
import numpy as np

from pilot_projection import ROOT, recovery_metrics, run_wave
from tv_regularization import value_and_gradient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pilot-budget', type=float, default=480.)
    parser.add_argument('--check-budget', type=float, default=120.)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = ['--sim-backend', 'slurm', '--host', 'HPC', '--partition', 'comp', '--account', 'energytech',
              '--qos', 'normal', '--nodes', '1', '--ranks', '1', '--cpus', '5', '--memory', '16',
              '--walltime', '00:01:00', '--concurrency', '5',
              '--binary-dir', '/cluster/home/krfo/adjoint-2d-v1', '--solver-tolerance', '1e-6']
    controls = ['--data', str(ROOT / 'runs/benchmark3-data-20261002/data.pkl'),
                '--variance', str(ROOT / 'runs/benchmark3-data-20261002/var.pkl'),
                '--ensemble-size', '250', '--opt-members', '5', '--seed', '2026',
                '--misfit-threshold', '100', '--single-update', '--prior-style', 'blocky',
                '--prior-projection', 'off']
    cases = {'baseline': ['--regularization', 'none'],
             'tv0': ['--regularization', 'tv', '--tv-weight', '0'],
             'tv005': ['--regularization', 'tv', '--tv-weight', '.05'],
             'tv020': ['--regularization', 'tv', '--tv-weight', '.2']}
    commands = []
    for zone in [27, 80]:
        for case, options in cases.items():
            name = f'{case}-{zone}'
            directory = output / name
            command = [sys.executable, '-B', '-u', str(ROOT / 'run_DA_sequential.py'),
                       '--output-dir', str(directory), '--start', str(zone), '--stop', str(zone + 3),
                       *common, *controls, *options]
            commands.append((name, command, directory, output / f'{name}.log'))
    report = {'commands': {name: command for name, command, _, _ in commands}, 'pilots': {},
              'forward_checks': {}, 'full_relaunch': False,
              'pilot_budget_seconds': args.pilot_budget, 'check_budget_seconds': args.check_budget}
    report['pilot_status'] = run_wave(commands, args.pilot_budget)
    # A fixed-fidelity single-point diagnostic exercises the 2D inner solver
    # even if every unforced pilot happens to pass the 1D gate.
    forced = []
    for case in ['tv0', 'tv005', 'tv020']:
        name = f'forced-{case}-80'
        directory = output / name
        command = [sys.executable, '-B', '-u', str(ROOT / 'run_DA_sequential.py'),
                   '--output-dir', str(directory), '--start', '80', '--stop', '81', '--force-2d',
                   *common, *controls, *cases[case]]
        forced.append((name, command, directory, output / f'{name}.log'))
    report['forced_commands'] = {name: command for name, command, _, _ in forced}
    report['forced_status'] = run_wave(forced, args.check_budget)
    with h5py.File(ROOT.parent / 'Dashboard/benchmarks/Benchmark-3/globalmodel.h5', 'r') as handle:
        truth = np.log(handle['formation/Rh'][:, handle['formation/Rh'].shape[1] // 2, :])
        x, z = handle['formation/x'][:, 0], handle['formation/z'][:, 0]
        wx = handle['wellpath/X'][:, 0]
    checks = []
    for name, _, directory, _ in commands + forced:
        paths = sorted(directory.glob('inversion_results_assim_*.pkl'),
                       key=lambda path: int(path.stem.rsplit('_', 1)[1]))
        rows = []
        check_path = None
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
                       tv_dose=result['tv_strength_applied'],
                       two_d_calls=result['two_d_calls'], substitutions=result['substituted_members'])
            rows.append(row)
            if result['selected_fidelity'] == '2D':
                check_path = path
        report['pilots'][name] = rows
        if check_path is not None:
            check_dir = output / f'check-{name}'
            command = [sys.executable, '-B', '-u', str(ROOT / 'check_posterior.py'),
                       '--checkpoint', str(check_path), '--output-dir', str(check_dir),
                       '--solve-timeout', '45', *common]
            checks.append((name, command, check_dir, output / f'check-{name}.log'))
    report['check_commands'] = {name: command for name, command, _, _ in checks}
    report['check_status'] = run_wave(checks, args.check_budget) if checks else {}
    for name, _, check_dir, _ in checks:
        if (check_dir / 'report.json').exists():
            report['forward_checks'][name] = json.loads((check_dir / 'report.json').read_text())
    (output / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Assessment: {output / "assessment.json"}')
    for name, rows in report['pilots'].items():
        print(name, len(rows), 'points')
        for row in rows:
            print(json.dumps(row))
    return 0 if all(status['returncode'] == 0 for status in report['pilot_status'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
