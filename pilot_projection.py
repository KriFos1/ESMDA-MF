"""Run bounded, matched projection on/off pilots and posterior forward checks."""

import argparse
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent


def cancel_own_jobs(run_dirs):
    jobs = set()
    for directory in run_dirs:
        for path in (directory / 'jobs').glob('*/handle.json'):
            handle = json.loads(path.read_text())
            job = str(handle.get('job_id', ''))
            if job and job.replace('_', '').isdigit():
                jobs.add(job)
    if jobs:
        subprocess.run(['ssh', 'HPC', 'scancel', *sorted(jobs)], timeout=30, check=False)


def run_wave(commands, budget):
    children = []
    started = time.monotonic()
    try:
        for name, command, run_dir, log in commands:
            stream = log.open('x')
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            children.append((name, process, stream, run_dir))
        while any(process.poll() is None for _, process, _, _ in children):
            if time.monotonic() - started > budget:
                break
            time.sleep(1.)
    finally:
        interrupted = [entry for entry in children if entry[1].poll() is None]
        for _, process, _, _ in interrupted:
            os.killpg(process.pid, signal.SIGTERM)
        for _, process, stream, _ in children:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            stream.close()
        if interrupted:
            cancel_own_jobs([entry[3] for entry in interrupted])
    timed_out = {entry[0] for entry in interrupted}
    return {name: {'returncode': process.returncode, 'timed_out': name in timed_out}
            for name, process, _, _ in children}


def recovery_metrics(result, truth, x, z, wx):
    nx, _, nz = result['dims']
    index = result['logging_index']
    column = int(np.argmin(abs(x - wx[index])))
    top = result['shift_ft']['z'] / 3.28084
    dz = result['spacing_ft'][2] / 3.28084
    target = np.array([np.mean(truth[(z >= top + k * dz) & (z < top + (k + 1) * dz), column])
                       for k in range(nz)])
    params = np.asarray(result['posterior_params'])
    rh = params[:, :nx*nz].reshape(-1, nx, nz)[:, nx//2, :] if result['selected_fidelity'] == '2D' \
        else params[:, :nz]
    log10_error = (rh - target) / np.log(10.)
    row = {'index': index, 'fidelity': result['selected_fidelity'],
           'rmse_log10': float(np.sqrt(np.mean(((rh.mean(0) - target) / np.log(10.)) ** 2))),
           'near_truth_fraction': float(np.mean(abs(log10_error) < .3)),
           'mean_max_jump_log10': float(np.max(abs(np.diff(rh, axis=1)), axis=1).mean() / np.log(10.)),
           'one_d_final_misfit': result['misfit']['1D_final'],
           'point_seconds': result['point_elapsed_seconds'],
           'posterior_prediction_kind': result['posterior_prediction_kind']}
    key = 'forward_posterior_misfit' if result['posterior_forward_verified'] else 'linearized_posterior_misfit'
    row[key] = float(np.mean(result['posterior_misfit']))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--points-per-zone', type=int, default=3)
    parser.add_argument('--pilot-budget', type=float, default=480.)
    parser.add_argument('--check-budget', type=float, default=120.)
    args = parser.parse_args(argv)
    if not 1 <= args.points_per_zone <= 3 or min(args.pilot_budget, args.check_budget) <= 0:
        parser.error('Use 1–3 points per zone and positive budgets')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = ['--sim-backend', 'slurm', '--host', 'HPC', '--partition', 'comp', '--account', 'energytech',
              '--qos', 'normal', '--nodes', '1', '--ranks', '1', '--cpus', '5', '--memory', '16',
              '--walltime', '00:01:00', '--concurrency', '5',
              '--binary-dir', '/cluster/home/krfo/adjoint-2d-v1', '--solver-tolerance', '1e-6']
    controls = ['--data', str(ROOT / 'runs/benchmark3-data-20261002/data.pkl'),
                '--variance', str(ROOT / 'runs/benchmark3-data-20261002/var.pkl'),
                '--ensemble-size', '250', '--opt-members', '5', '--seed', '2026',
                '--misfit-threshold', '100', '--single-update', '--prior-style', 'blocky']
    commands = []
    for zone in [27, 80]:
        for projection in ['off', 'on']:
            name = f'{projection}-{zone}'
            directory = output / name
            command = [sys.executable, '-B', '-u', str(ROOT / 'run_DA_sequential.py'),
                       '--output-dir', str(directory), '--start', str(zone),
                       '--stop', str(zone + args.points_per_zone), '--prior-projection', projection,
                       *common, *controls]
            commands.append((name, command, directory, output / f'{name}.log'))
    report = {'configuration': vars(args) | {'output_dir': str(output)},
              'commands': {name: command for name, command, _, _ in commands},
              'prior_units': 'natural_log internally; CLI log10', 'pilots': {}, 'forward_checks': {}}
    report['pilot_status'] = run_wave(commands, args.pilot_budget)
    benchmark = ROOT.parent / 'Dashboard/benchmarks/Benchmark-3'
    with h5py.File(benchmark / 'globalmodel.h5', 'r') as handle:
        truth = np.log(handle['formation/Rh'][:, handle['formation/Rh'].shape[1] // 2, :])
        x = handle['formation/x'][:, 0]
        z = handle['formation/z'][:, 0]
        wx = handle['wellpath/X'][:, 0]
    checks = []
    for name, _, directory, _ in commands:
        paths = sorted(directory.glob('inversion_results_assim_*.pkl'),
                       key=lambda path: int(path.stem.rsplit('_', 1)[1]))
        rows = []
        check_path = None
        for path in paths:
            result = pickle.loads(path.read_bytes())
            rows.append(recovery_metrics(result, truth, x, z, wx))
            if result['selected_fidelity'] == '2D':
                check_path = path
        report['pilots'][name] = rows
        if check_path is not None:
            check_dir = output / f'check-{name}'
            command = [sys.executable, '-B', '-u', str(ROOT / 'check_posterior.py'),
                       '--checkpoint', str(check_path), '--output-dir', str(check_dir),
                       '--solve-timeout', '45', *common]
            checks.append((name, command, check_dir, output / f'check-{name}.log'))
    report['check_status'] = run_wave(checks, args.check_budget) if checks else {}
    for name, _, check_dir, _ in checks:
        path = check_dir / 'report.json'
        if path.exists():
            report['forward_checks'][name] = json.loads(path.read_text())
    report['full_relaunch'] = False
    (output / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Assessment: {output / "assessment.json"}')
    for name, rows in report['pilots'].items():
        print(name, report['pilot_status'][name], 'checkpoints', len(rows))
        for row in rows:
            print(json.dumps(row))
    return 0 if all(status['returncode'] == 0 for status in report['pilot_status'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
