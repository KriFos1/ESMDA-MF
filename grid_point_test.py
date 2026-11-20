#!/usr/bin/env python3
"""What does vertical resolution buy in the 2D update? One gated point re-solved on 5 ft and on 1 ft rows.

graded_point_test.py could not answer this: its prior was the sparse precision fitted to the carried
ensemble, whose 1-ft rows inside a former 5 ft cell are copies, so no boundary could move inside a cell.
Here both grids get the same kind of prior, one that carries no 5 ft structure in its correlations: the
Matern prior with the chain's ranges in feet, around the carried ensemble's cellwise mean and (floored) std
(``prior_type = matern_cellwise``). Same data, covariance and merged logging points as the chain, start at
the carried mean, same iteration cap.

  5ft   the chain's rows, 80 of 5 ft
  1ft   400 rows of 1 ft (the carried ensemble copied onto them); the Matern prior is the same in feet

Both posteriors, and the chain's own, are scored on 1-ft rows against the benchmark upscaled onto 1-ft rows
(layered upscaling); classes are those of the 5 ft truth (boundary_check.classify). The chain is not touched.

python -B -u grid_point_test.py --chain runs/fault7-merge3-g25-s100-starts0-it10 --index 126
python -B -u grid_point_test.py --chain runs/fault7-merge3-g25-s100-starts0-it10 --index 126 --score-only
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

import run_chain_laplace_loop as loop
from chain_laplace import laplace_posterior, load_problem
from multistart_point import benchmark_window, resample_rows, resample_state
from refine_z_check import partitions

ROOT = Path(__file__).resolve().parent
LN10 = np.log(10.)


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--grids', nargs='+', default=['5ft', '1ft'], choices=['5ft', '1ft'])
    parser.add_argument('--label', default='merge3')
    parser.add_argument('--samples', type=int, default=250)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--source', type=Path, default=ROOT / 'references/map-source')
    parser.add_argument('--binary-dir', default='/cluster/home/mlie/cluster-build')
    parser.add_argument('--orientation', default='bedding')
    parser.add_argument('--host', default='HPC')
    parser.add_argument('--hpc-root', default='/cluster/work/users/mlie/esmda-runs')
    parser.add_argument('--hpc-python', default='/cluster/home/mlie/esmda-runs/.venv/bin/python')
    parser.add_argument('--poll', type=float, default=60.)
    parser.add_argument('--optimizer-setting', action='append', metavar='KEY=VALUE',
                        default=['stagnation_tolerance=0.01', 'stagnation_iterations=1', 'max_rounds=2',
                                 'max_iterations=10'])
    args = parser.parse_args(argv)
    chain, index = args.chain.resolve(), args.index
    source = chain / f'laplace-problem-{index:03d}.npz'
    problem = load_problem(source)
    nx, ny, nz = problem['dims']
    counts = partitions(problem)
    rows = len(counts['1ft'])

    runs = {}
    for grid in args.grids:
        name = f'grid-idx{index:03d}-{args.label}-{grid}'
        run_dir, remote = ROOT / 'runs' / 'point-tests' / 'grid' / name, f'{args.hpc_root}/{name}'
        problem_file = ROOT / 'runs' / 'point-tests' / 'grid' / f'{name}-problem.npz'
        if not problem_file.exists():
            with np.load(source) as saved:
                values = {key: saved[key].copy() for key in saved.files}
            values['prior_type'] = np.array('matern_cellwise')
            if grid == '1ft':
                values['prior_ensemble'] = resample_state(values['prior_ensemble'], nx,
                                                          resample_rows(counts['5ft'], counts['1ft']))
                values['dims'] = np.array([nx, ny, rows])
                values['spacing'] = np.array([problem['spacing'][0], problem['spacing'][1], 1.])
                values['graded_dz'] = np.asarray(counts['1ft'])
            np.savez_compressed(problem_file, **values)
        job = None
        if not args.score_only:
            if not run_dir.exists():
                loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'prepare-laplace', '--problem',
                          problem_file, '--source', args.source, '--run-dir', run_dir, '--binary-dir',
                          args.binary_dir, '--orientation', args.orientation,
                          *(item for setting in args.optimizer_setting for item in ('--optimizer-setting', setting))])
                loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'bundle', '--run-dir', run_dir])
                loop.ssh(args.host, f'mkdir -p {remote}')
                loop.retry(lambda: loop.run(['scp', '-q', run_dir / 'deployment.tar.gz', f'{args.host}:{remote}/']))
                loop.ssh(args.host, f'cd {remote} && tar xzf deployment.tar.gz && rm deployment.tar.gz')
            if loop.select_member(run_dir, 1) is None or not (run_dir / 'member-000' / 'laplace.npz').exists():
                job = loop.active_job(args.host, f'{remote}/run')
                if job is None:
                    job = loop.ssh(args.host, f'cd {remote} && export RML_PROJECT={remote}/code RML_RUN={remote}/run '
                                              f'RML_PYTHON={args.hpc_python} && sbatch --parsable --export=ALL '
                                              f'--array=0 --chdir=$RML_RUN $RML_RUN/rml-array.sbatch',
                                   capture=True).strip()
                    loop.log(f'{name}: job {job} submitted')
        runs[grid] = (run_dir, remote, problem_file, job)
    for grid, (run_dir, remote, _, job) in runs.items():
        if job is None:
            continue
        loop.wait_for_array(args.host, job, [0], args.poll)
        loop.retry(lambda: loop.run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'deployment.tar.gz',
                                     '--exclude', 'inputs.npz', '--exclude', 'study.json', '--exclude',
                                     'prior-*.npz', f'{args.host}:{remote}/run/', f'{run_dir}/']))
        loop.ssh(args.host, f'rm -rf {remote}/run/member-000/jobs {remote}/run/member-000/checkpoint.npz')

    with open(chain / f'inversion_results_assim_{index}.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    cells = nx * nz
    compare = load_module('compare_runs', 'compare-runs.py')
    boundary = load_module('boundary_check', 'boundary_check.py')
    coarse = compare.Truth(json.loads((chain / 'study.json').read_text())['benchmark']).window(saved)['rh']
    edges = problem['shift_ft']['z'] + np.arange(rows + 1)
    fine = benchmark_window(chain, {**problem, 'dims': [nx, ny, rows]}, edges)[:nx * rows].reshape(nx, rows) / LN10
    kind = np.repeat(boundary.classify(coarse), counts['5ft'], axis=1)
    dx = saved['spacing_ft'][0]
    xo, zo = (np.arange(nx) - nx // 2) * dx, np.arange(rows) + .5 - rows / 2
    zone = np.outer(np.abs(xo) <= compare.NEAR_X_FT, np.abs(zo) <= compare.ZONE_Z_FT)

    def expanded(state, partition):
        field = np.asarray(state)[:nx * len(partition)].reshape(nx, len(partition), -1) / LN10
        return np.repeat(field, partition, axis=1)

    methods = {'chain (precision)': expanded(np.asarray(saved['posterior_ensemble_params']).T, counts['5ft'])}
    print('MAP fit of the gate point\'s own data, misfit per datum against the measurement covariance')
    for grid, (run_dir, _, problem_file, _) in runs.items():
        if not run_dir.exists() or loop.select_member(run_dir, 1) is None:
            print(f'  {grid}: not finished')
            continue
        out = laplace_posterior(chain, problem_file, run_dir, args.samples, args.seed)
        np.savez_compressed(run_dir / 'posterior.npz', models=out['models'], map=out['model'])
        methods[f'Matern, {grid} rows'] = expanded(out['models'], counts[grid])
        result = out['result']
        print(f'  {grid}: {out["map_misfit"]:.2f}; {result.get("accepted_iterations")} iterations, '
              f'{result.get("reason")}, {result.get("seconds_this_attempt", float("nan")):.0f} s')
    print('\nnear bit (log10 Rh, on 1-ft rows against the truth on 1-ft rows)')
    print('  method               class                  rows    error  spread  z-rms  cover90')
    for label, field in methods.items():
        mean, std = field.mean(axis=2), field.std(axis=2, ddof=1)
        inside = (np.quantile(field, .05, axis=2) <= fine) & (fine <= np.quantile(field, .95, axis=2))
        err = mean - fine
        for text, m in [('all cells', zone)] + [(boundary.CLASSES[k], zone & (kind == k)) for k in range(5)]:
            if m.any():
                print(f'  {label:20s} {text:21s} {m.sum() / zone.sum():6.1%} {np.sqrt(np.mean(err[m] ** 2)):7.3f} '
                      f'{std[m].mean():7.3f} {np.sqrt(np.mean((err / std)[m] ** 2)):6.2f} {inside[m].mean():8.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
