#!/usr/bin/env python3
"""Finer rows at the tool in the 2D update: one gated point re-solved on a graded vertical grid.

The chain's 2D problem at the point (merged logging points included, same data and covariance, start at
the carried mean) is solved again on a graded grid with the same number of rows: 1 ft at the tool, growing
away from it (multistart_point.graded_rows). The carried ensemble is averaged onto the graded rows and the
same kind of prior is fitted to it. The forward solve costs the same (refine_z_check.py).

Both posteriors are scored on 1-ft rows against the benchmark upscaled onto 1-ft rows (layered upscaling):
the chain's is expanded from its 5 ft rows, the graded one from its own rows. A 5 ft cell that holds a bed
boundary cannot match the 1-ft truth on both sides, so this reference shows what resolution buys; the
chain's score against the truth on its own 5 ft rows is printed next to it. Classes are those of the 5 ft
truth (boundary_check.classify), so the same rows are compared. The chain is not touched.

python -B -u graded_point_test.py --chain runs/fault7-merge3-g25-s100-starts0-it10 --index 126
python -B -u graded_point_test.py --chain runs/fault7-merge3-g25-s100-starts0-it10 --index 126 --score-only
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
    name = f'graded-idx{index:03d}-{args.label}'
    run_dir, remote = ROOT / 'runs' / 'point-tests' / 'graded' / name, f'{args.hpc_root}/{name}'
    problem_file = ROOT / 'runs' / 'point-tests' / 'graded' / f'{name}-problem.npz'
    if not problem_file.exists():
        with np.load(source) as saved:
            values = {key: saved[key].copy() for key in saved.files}
        values['prior_ensemble'] = resample_state(values['prior_ensemble'], nx,
                                                  resample_rows(counts['5ft'], counts['graded']))
        values['graded_dz'] = np.asarray(counts['graded'])
        np.savez_compressed(problem_file, **values)
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
            loop.wait_for_array(args.host, job, [0], args.poll)
            loop.retry(lambda: loop.run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'deployment.tar.gz',
                                         '--exclude', 'inputs.npz', '--exclude', 'study.json', '--exclude',
                                         'prior-precision.npz', f'{args.host}:{remote}/run/', f'{run_dir}/']))
            loop.ssh(args.host, f'rm -rf {remote}/run/member-000/jobs {remote}/run/member-000/checkpoint.npz')
    if loop.select_member(run_dir, 1) is None:
        print('not finished')
        return 1

    with open(chain / f'inversion_results_assim_{index}.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    out = laplace_posterior(chain, problem_file, run_dir, args.samples, args.seed)
    np.savez_compressed(run_dir / 'posterior.npz', models=out['models'], map=out['model'],
                        graded_dz=np.asarray(counts['graded']))
    print(f'graded rows (1-ft units, top to bottom): {counts["graded"]}')
    print(f'MAP misfit of the gate point\'s own data: {out["map_misfit"]:.2f} per datum; '
          f'{out["result"].get("accepted_iterations")} iterations, {out["result"].get("reason")}, '
          f'{out["result"].get("seconds_this_attempt", float("nan")):.0f} s')

    cells, rows = nx * nz, len(counts['1ft'])
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
        field = np.asarray(state)[:cells].reshape(nx, nz, -1) / LN10
        return np.repeat(field, partition, axis=1)

    chain_field = expanded(np.asarray(saved['posterior_ensemble_params']).T, counts['5ft'])
    methods = {'chain, 5 ft rows': (chain_field, fine), 'graded rows': (expanded(out['models'], counts['graded']), fine),
               'chain vs 5 ft truth': (chain_field, np.repeat(coarse, counts['5ft'], axis=1))}
    print('\nnear bit (log10 Rh, on 1-ft rows)')
    print('  method               class                  rows    error  spread  z-rms  cover90')
    for label, (field, truth) in methods.items():
        mean, std = field.mean(axis=2), field.std(axis=2, ddof=1)
        inside = (np.quantile(field, .05, axis=2) <= truth) & (truth <= np.quantile(field, .95, axis=2))
        err = mean - truth
        for text, m in [('all cells', zone)] + [(boundary.CLASSES[k], zone & (kind == k)) for k in range(5)]:
            if m.any():
                print(f'  {label:20s} {text:21s} {m.sum() / zone.sum():6.1%} {np.sqrt(np.mean(err[m] ** 2)):7.3f} '
                      f'{std[m].mean():7.3f} {np.sqrt(np.mean((err / std)[m] ** 2)):6.2f} {inside[m].mean():8.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
