#!/usr/bin/env python3
"""What would finer rows buy, and what do they cost? (forward only)

The 2D solver has its own fixed node layout (about 2 ft at the tool), so the rows of the inversion grid
decide what model it is handed, not how long a solve takes. At chosen gate points of a chain the benchmark
truth is averaged onto the window with different row partitions and run through the 2D solver at the gate
point's tool position. The misfit of that response to the measured data, against the measurement
covariance, is the model error of the grid: what no inversion on that grid can fit away.

  5ft      the chain's grid, 80 rows of 5 ft (control: must reproduce truth_forward_2d.py)
  2.5ft    160 rows, alternating 2 and 3 ft (the solver takes whole feet)
  1ft      400 rows of 1 ft
  graded   80 rows as today, 1 ft at the tool and growing away (multistart_point.graded_rows)

Rows are handed to the solver as 1-ft rows (``graded_dz``), also for the 5 ft control.

python -B -u refine_z_check.py --chain runs/fault4-control-g25-s100 --indices 115 120 126 132
python -B -u refine_z_check.py --chain runs/fault4-control-g25-s100 --indices 115 120 126 132 --report-only
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

import run_chain_laplace_loop as loop
from chain_laplace import load_problem
from multistart_point import benchmark_window, graded_rows, resample_rows, resample_state

ROOT = Path(__file__).resolve().parent
LN10 = np.log(10.)


def partitions(problem):
    """Row counts in 1-ft rows for every grid variant of the problem's window."""
    nz, dz = problem['dims'][2], problem['spacing'][2]
    rows = int(round(nz * dz))
    tool_tvd = float(np.asarray(problem['well_Z']).ravel()[problem['index']]) * 3.28084
    tool_row = int(np.floor(tool_tvd - problem['shift_ft']['z']))
    return {'5ft': [int(round(dz))] * nz, '2.5ft': [2, 3] * (rows // 5), '1ft': [1] * rows,
            'graded': [int(n) for n in graded_rows(rows, nz, tool_row)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chain', type=Path, required=True, help='Chain with single-point 2D problems')
    parser.add_argument('--indices', type=int, nargs='+', required=True)
    parser.add_argument('--grids', nargs='+', default=['5ft', '2.5ft', '1ft', 'graded'])
    parser.add_argument('--upscaling', choices=['arithmetic', 'layered'], default='arithmetic',
                        help='arithmetic: Rh and Rv averaged (as the scoring truth); layered: Rh harmonic, Rv '
                             'arithmetic (the effective medium of thin beds, multistart_point.benchmark_window)')
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

    tag = '' if args.upscaling == 'arithmetic' else f'-{args.upscaling}'

    def directory(index, grid):
        return ROOT / 'runs' / 'point-tests' / 'refine-z' / f'refine-z-idx{index:03d}-{grid}{tag}'

    jobs = {}
    for index in args.indices:
        source = chain / f'laplace-problem-{index:03d}.npz'
        problem = load_problem(source)
        if 'merge_channels' in problem:
            raise ValueError('Use a chain with single-point 2D problems')
        nx, ny, nz = problem['dims']
        with open(chain / f'inversion_results_assim_{index}.pkl', 'rb') as stream:
            saved = pickle.load(stream)
        counts = partitions(problem)
        rows = len(counts['1ft'])
        # Truth on 1-ft rows (resistivities averaged, as on the chain's grid), then onto each partition.
        fine = truth_model.window({**saved, 'dims': [nx, ny, rows], 'spacing_ft': [saved['spacing_ft'][0],
                                                                                   saved['spacing_ft'][1], 1.]})
        resistivity = {key: 10. ** fine[key] for key in ('rh', 'rv')}
        for grid in args.grids:
            run_dir = directory(index, grid)
            remote = f'{args.hpc_root}/{run_dir.name}'
            if (run_dir / 'member-000' / 'forward.npz').exists() or args.report_only:
                continue
            if not run_dir.exists():
                average = resample_rows(counts['1ft'], counts[grid])
                rh, rv = (np.log(resistivity[key] @ average.T) for key in ('rh', 'rv'))
                truth = np.concatenate([rh.ravel(), (rv - rh).ravel()])
                if args.upscaling == 'layered':
                    edges = problem['shift_ft']['z'] + np.r_[0, np.cumsum(counts[grid])]
                    truth = benchmark_window(chain, {**problem, 'dims': [nx, ny, len(counts[grid])]}, edges)
                with np.load(source) as stored:
                    values = {key: stored[key].copy() for key in stored.files}
                # The prior is not used by a forward run; it only has to match the grid.
                values['prior_ensemble'] = resample_state(values['prior_ensemble'], nx,
                                                          resample_rows(counts['5ft'], counts[grid]))
                values['dims'] = np.array([nx, ny, len(counts[grid])])
                values['graded_dz'] = np.asarray(counts[grid])
                values['prior_type'] = np.array('matern_cellwise')
                problem_file = ROOT / 'runs' / 'point-tests' / 'refine-z' / f'{run_dir.name}-problem.npz'
                np.savez_compressed(problem_file, **values)
                start = ROOT / 'runs' / 'point-tests' / 'refine-z' / f'{run_dir.name}-start.npz'
                np.savez(start, models=truth[None, :], labels=np.array([f'truth on the {grid} rows']))
                loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'prepare-laplace', '--problem',
                          problem_file, '--source', args.source, '--run-dir', run_dir, '--binary-dir',
                          args.binary_dir, '--orientation', args.orientation, '--start-models', start])
                loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'bundle', '--run-dir', run_dir])
                loop.ssh(args.host, f'mkdir -p {remote}')
                loop.retry(lambda: loop.run(['scp', '-q', run_dir / 'deployment.tar.gz', f'{args.host}:{remote}/']))
                loop.ssh(args.host, f'cd {remote} && tar xzf deployment.tar.gz && rm deployment.tar.gz')
            job = loop.active_job(args.host, f'{remote}/run')
            if job is None:
                job = loop.ssh(args.host, f'cd {remote} && export RML_PROJECT={remote}/code RML_RUN={remote}/run '
                                          f'RML_PYTHON={args.hpc_python} RML_ACTION=forward-start && sbatch '
                                          f'--parsable --export=ALL --array=0 --time=00:20:00 --chdir=$RML_RUN '
                                          f'$RML_RUN/rml-array.sbatch', capture=True).strip()
                loop.log(f'index {index}, {grid}: forward job {job} submitted')
            jobs[index, grid] = (job, remote, run_dir)
    for (index, grid), (job, remote, run_dir) in jobs.items():
        loop.wait_for_array(args.host, job, [0], args.poll)
        loop.retry(lambda: loop.run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'inputs.npz', '--exclude',
                                     'study.json', '--exclude', 'prior-*.npz', f'{args.host}:{remote}/run/',
                                     f'{run_dir}/']))

    print(f'\nMisfit of the truth ({args.upscaling} upscaling) per datum against the measurement covariance '
          '(forward seconds)')
    print('  index  ' + ' '.join(f'{grid:>18s}' for grid in args.grids))
    for index in args.indices:
        cells = []
        for grid in args.grids:
            member = directory(index, grid) / 'member-000'
            if not (member / 'forward.npz').exists():
                cells.append(f'{"-":>18s}')
                continue
            with np.load(directory(index, grid) / 'inputs.npz') as inputs:
                observations, variance = inputs['observations'], inputs['variance']
            misfit = float(np.mean((np.load(member / 'forward.npz')['prediction'] - observations) ** 2 / variance))
            seconds = json.loads((member / 'forward.json').read_text())['seconds']
            cells.append(f'{misfit:10.1f} ({seconds:4.0f} s)')
        print(f'  {index:5d}  ' + ' '.join(cells))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
