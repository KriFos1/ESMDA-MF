#!/usr/bin/env python3
"""2D model error at chosen logging points: 2D response of the truth on the inversion grid against the data.

For each index one forward-only job runs on the HPC for two models: the benchmark truth averaged onto the
5 ft window grid, and (as a control) the chain's own posterior model at that point. The misfit of the
truth, in units of the data error model, is the 2D model error the inversion does not account for.
Gated points use the chain's exported problem; for the other points an equivalent problem is assembled
from the chain's checkpoint (same data, covariance, mask and window).

python -B -u truth_forward_2d.py --chain runs/my-run-4 --indices 10 45 60 80 95 125 150 170 178
python -B -u truth_forward_2d.py --chain runs/my-run-4 --indices 10 45 --report-only
"""
import argparse
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_chain_laplace_loop as loop
from esmda_remote2d import UDAR_COMPONENTS

ROOT = Path(__file__).resolve().parent
LN10 = np.log(10.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'compare-output/model-error')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT / 'references/map-source')
    parser.add_argument('--binary-dir', default='/cluster/home/mlie/cluster-build')
    parser.add_argument('--orientation', default='bedding')
    parser.add_argument('--host', default='HPC')
    parser.add_argument('--hpc-root', default='/cluster/work/users/mlie/esmda-runs')
    parser.add_argument('--hpc-python', default='/cluster/home/mlie/esmda-runs/.venv/bin/python')
    parser.add_argument('--poll', type=float, default=45.)
    args = parser.parse_args(argv)
    chain = args.chain.resolve()
    spec = importlib.util.spec_from_file_location('compare_runs', ROOT / 'compare-runs.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    truth_model = compare.Truth(json.loads((chain / 'study.json').read_text())['benchmark'])
    template = sorted(chain.glob('laplace-problem-[0-9][0-9][0-9].npz'))[0]

    def run_dir(index):
        return ROOT / 'runs' / 'point-tests' / 'truth-fwd' / f'truth-fwd-idx{index:03d}'

    def done(index):
        return all((run_dir(index) / f'member-{member:03d}' / 'forward.npz').exists() for member in (0, 1))

    jobs = {}
    for index in args.indices:
        if done(index) or args.report_only:
            continue
        directory, remote = run_dir(index), f'{args.hpc_root}/truth-fwd-idx{index:03d}'
        with open(chain / f'inversion_results_assim_{index}.pkl', 'rb') as stream:
            saved = pickle.load(stream)
        if not directory.exists():
            problem = chain / f'laplace-problem-{index:03d}.npz'
            if not problem.exists():
                # Not a gated point: same fields as an exported problem, filled from the checkpoint.
                with np.load(template) as source:
                    values = {key: source[key].copy() for key in source.files}
                shift = saved['shift_ft']
                values.update(index=np.int64(index), prior_ensemble=np.asarray(saved['prior_params'], float),
                              observations=np.asarray(saved['observations']),
                              variance=np.asarray(saved['data_covariance']),
                              mask=np.asarray(saved['observation_mask'], bool),
                              shift=np.array([shift['x'], shift['y'], shift['z']], float),
                              movement=np.asarray(saved['window_movement'], int),
                              probe=np.array(json.dumps({'note': 'assembled for a forward-only check'})))
                problem = ROOT / 'runs' / 'point-tests' / 'truth-fwd' / f'truth-fwd-idx{index:03d}-problem.npz'
                np.savez_compressed(problem, **values)
            target = truth_model.window(saved)
            truth = np.concatenate([target['rh'].ravel(), target['ratio'].ravel()]) * LN10
            start = ROOT / 'runs' / 'point-tests' / 'truth-fwd' / f'truth-fwd-idx{index:03d}-start.npz'
            # 2D points store the MAP (1 x n), 1D points the ensemble (members x n): use its mean.
            posterior = np.asarray(saved['posterior_params'], float).mean(axis=0)
            np.savez(start, models=np.vstack([truth, posterior]),
                     labels=np.array(['truth on the 5 ft grid', 'chain posterior model']))
            loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'prepare-laplace', '--problem', problem,
                      '--source', args.source, '--run-dir', directory, '--binary-dir', args.binary_dir,
                      '--orientation', args.orientation, '--start-models', start])
            loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'bundle', '--run-dir', directory])
            loop.ssh(args.host, f'mkdir -p {remote}')
            loop.retry(lambda: loop.run(['scp', '-q', directory / 'deployment.tar.gz', f'{args.host}:{remote}/']))
            loop.ssh(args.host, f'cd {remote} && tar xzf deployment.tar.gz && rm deployment.tar.gz')
        job = loop.active_job(args.host, f'{remote}/run')
        if job is None:
            job = loop.ssh(args.host, f'cd {remote} && export RML_PROJECT={remote}/code RML_RUN={remote}/run '
                                      f'RML_PYTHON={args.hpc_python} RML_ACTION=forward-start && sbatch --parsable '
                                      f'--export=ALL --array=0-1 --time=00:20:00 --chdir=$RML_RUN '
                                      f'$RML_RUN/rml-array.sbatch', capture=True).strip()
            loop.log(f'index {index}: forward job {job} submitted')
        jobs[index] = (job, remote)
    while jobs:
        time.sleep(args.poll)
        queue = loop.ssh(args.host, "squeue -u $USER -h -o '%A'", capture=True).split()
        for index, (job, remote) in list(jobs.items()):
            if job.split('_')[0] not in queue:
                loop.retry(lambda: loop.run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'inputs.npz',
                                             '--exclude', 'study.json', '--exclude', 'prior-precision.npz',
                                             f'{args.host}:{remote}/run/', f'{run_dir(index)}/']))
                loop.log(f'index {index}: {"done" if done(index) else "FAILED (see the member logs)"}')
                del jobs[index]

    history = json.loads((chain / 'history.json').read_text())
    gated = {record['logging_index'] for record in history if record.get('stage') is not None}
    rows, channels = [], []
    for directory in sorted((ROOT / 'runs' / 'point-tests' / 'truth-fwd').glob('truth-fwd-idx[0-9][0-9][0-9]')):
        index = int(directory.name[-3:])
        if not done(index):
            continue
        with np.load(directory / 'inputs.npz') as inputs:
            observations, variance, mask = inputs['observations'], inputs['variance'], inputs['mask']
        tools = json.loads((directory / 'study.json').read_text())['tools']
        residual = [(np.load(directory / f'member-{member:03d}' / 'forward.npz')['prediction'] - observations)
                    / np.sqrt(variance) for member in (0, 1)]
        rows.append({'index': index, 'stage': '2D' if index in gated else '1D', 'channels': len(observations),
                     'truth_2d': float(np.mean(residual[0] ** 2)), 'chain_model_2d': float(np.mean(residual[1] ** 2))})
        for value, position in zip(residual[0], np.flatnonzero(mask)):
            channels.append({'index': index, 'tool': '/'.join(tools[position // len(UDAR_COMPONENTS)]),
                             'component': UDAR_COMPONENTS[position % len(UDAR_COMPONENTS)], 'truth_2d': float(value)})
    table, detail = pd.DataFrame(rows), pd.DataFrame(channels)
    args.out.mkdir(parents=True, exist_ok=True)
    one_d = args.out / 'points.csv'
    if one_d.exists():
        table = table.merge(pd.read_csv(one_d)[['index', 'data_coarse', 'data_fine']].rename(
            columns={'data_coarse': 'truth_1d', 'data_fine': 'truth_1d_fine_grid'}), on='index', how='left')
    table.to_csv(args.out / 'points_2d.csv', index=False)
    detail.to_csv(args.out / 'channels_2d.csv', index=False)
    print('\nMisfit per datum in units of the data error variance')
    print(table.round(1).to_string(index=False))
    rms = lambda values: np.sqrt(np.mean(values ** 2))
    print('\nTruth, 2D: root mean squared mismatch in units of the data error')
    print('  by component:', detail.groupby('component')['truth_2d'].agg(rms).reindex(UDAR_COMPONENTS).round(1).to_dict())
    print('  by tool:', detail.groupby('tool')['truth_2d'].agg(rms).round(1).to_dict())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
