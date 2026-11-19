#!/usr/bin/env python3
"""Can earlier logging points be simulated inside the window of the current one? (forward only)

For merging several logging points in one 2D update, the tool is placed at earlier points (e.g. 25 and
50 ft behind the bit) inside the window of the current point. This checks what that costs: the 2D response
of the truth on the current window with the tool at an earlier point is compared with the data measured
there, next to the same comparison in that point's own window (truth_forward_2d.py). The difference is the
model error added by the window ending closer to the tool on the trailing side. The chain's MAP at the
current point is run as a second model: its misfit to the earlier data shows whether they add information.

python -B -u merged_points_check.py --chain runs/fault4-control-g25-s100 --index 115 --behind 110 105
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

import run_chain_laplace_loop as loop
from esmda_core import align_phases, phase_channels
from esmda_remote2d import UDAR_COMPONENTS

ROOT = Path(__file__).resolve().parent
LN10 = np.log(10.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--behind', type=int, nargs='+', required=True)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT / 'references/map-source')
    parser.add_argument('--binary-dir', default='/cluster/home/mlie/cluster-build')
    parser.add_argument('--orientation', default='bedding')
    parser.add_argument('--host', default='HPC')
    parser.add_argument('--hpc-root', default='/cluster/work/users/mlie/esmda-runs')
    parser.add_argument('--hpc-python', default='/cluster/home/mlie/esmda-runs/.venv/bin/python')
    parser.add_argument('--poll', type=float, default=45.)
    args = parser.parse_args(argv)
    chain, index = args.chain.resolve(), args.index
    spec = importlib.util.spec_from_file_location('compare_runs', ROOT / 'compare-runs.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    truth_model = compare.Truth(json.loads((chain / 'study.json').read_text())['benchmark'])

    def checkpoint(point):
        with open(chain / f'inversion_results_assim_{point}.pkl', 'rb') as stream:
            return pickle.load(stream)

    current = checkpoint(index)
    target = truth_model.window(current)
    models = np.vstack([np.concatenate([target['rh'].ravel(), target['ratio'].ravel()]) * LN10,
                        np.asarray(current['posterior_params'], float).mean(axis=0)])
    labels = ['truth on the window grid', f'chain MAP at {index}']
    jobs = {}
    for tool in [index, *args.behind]:
        name = f'merged-fwd-w{index:03d}-t{tool:03d}'
        directory, remote = ROOT / 'runs' / 'point-tests' / 'merged-fwd' / name, f'{args.hpc_root}/{name}'
        if all((directory / f'member-{m:03d}' / 'forward.npz').exists() for m in (0, 1)) or args.report_only:
            continue
        if not directory.exists():
            start = ROOT / 'runs' / 'point-tests' / 'merged-fwd' / f'{name}-start.npz'
            np.savez(start, models=models, labels=np.array(labels))
            loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'prepare-laplace', '--problem',
                      chain / f'laplace-problem-{index:03d}.npz', '--source', args.source, '--run-dir', directory,
                      '--binary-dir', args.binary_dir, '--orientation', args.orientation, '--start-models', start])
            # Same window and model, tool at another logging point (the solver places it by its coordinates).
            study = json.loads((directory / 'study.json').read_text())
            study.update(index=tool, tool_note=f'window of index {index}, tool at index {tool}')
            (directory / 'study.json').write_text(json.dumps(study, indent=2) + '\n')
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
            loop.log(f'{name}: forward job {job} submitted')
        jobs[tool] = (job, remote, directory)
    for tool, (job, remote, directory) in jobs.items():
        loop.wait_for_array(args.host, job, [0, 1], args.poll)
        loop.retry(lambda: loop.run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'inputs.npz', '--exclude',
                                     'study.json', '--exclude', 'prior-precision.npz',
                                     f'{args.host}:{remote}/run/', f'{directory}/']))

    print(f'\nMisfit per datum against the measurement covariance, model on the window of index {index}')
    print('  tool at   offset ft   channels   ' + '   '.join(f'{label:>26s}' for label in labels))
    md = None
    for tool in [index, *args.behind]:
        directory = ROOT / 'runs' / 'point-tests' / 'merged-fwd' / f'merged-fwd-w{index:03d}-t{tool:03d}'
        saved = checkpoint(tool)
        mask = np.asarray(saved['observation_mask'], bool)
        observations, variance = np.asarray(saved['observations']), np.asarray(saved['data_covariance'])
        phase = phase_channels(mask, UDAR_COMPONENTS)
        values = []
        for member in (0, 1):
            path = directory / f'member-{member:03d}' / 'forward.npz'
            if not path.exists():
                values.append(np.nan)
                continue
            prediction = align_phases(np.load(path)['prediction_full'][mask], observations, phase)
            values.append(float(np.mean((prediction - observations) ** 2 / variance)))
        offset = saved['shift_ft']['x'] - current['shift_ft']['x']
        print(f'  {tool:7d} {offset:11.1f} {int(mask.sum()):10d}   ' + '   '.join(f'{value:26.1f}' for value in values))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
