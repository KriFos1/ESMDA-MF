#!/usr/bin/env python3
"""A model-error term of the right size in 2D: one gated point re-solved with the data std times a factor.

With layered upscaling the truth on the 5 ft grid misfits the data by 3-9 per datum (refine_z_check.py), so
the grid's own error is about 1.5-3 measurement standard deviations. The chain's 2D problem at the point
(merged logging points included, same carried prior, start at the carried mean) is solved again with every
variance multiplied by factor^2; the Laplace posterior is scored by distance to the truth boundaries, next
to the carried prior and the chain's posterior. The earlier test (model_error_2d_point.py) sized the term
from the arithmetic truth and was several times too large. The chain is not touched.

python -B -u model_error_2d_flat.py --chain runs/fault7-merge3-g25-s100-starts0-it10 --index 126 --factors 2 3
python -B -u model_error_2d_flat.py --chain runs/fault7-merge3-g25-s100-starts0-it10 --index 126 --score-only
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

import run_chain_laplace_loop as loop
from chain_laplace import laplace_posterior, load_problem

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
    parser.add_argument('--factors', type=float, nargs='+', default=[2., 3.], help='Factors on the data std')
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
    if json.loads((chain / 'study.json').read_text()).get('model_error', 'none') != 'none':
        raise ValueError('Use a chain without a model-error term: its problem holds the measurement covariance')

    runs = {}
    for factor in args.factors:
        name = f'me2d-flat-idx{index:03d}-{args.label}-x{factor:g}'
        run_dir, remote = ROOT / 'runs' / 'point-tests' / 'me2d-flat' / name, f'{args.hpc_root}/{name}'
        problem_file = ROOT / 'runs' / 'point-tests' / 'me2d-flat' / f'{name}-problem.npz'
        if not problem_file.exists():
            with np.load(source) as saved:
                values = {key: saved[key].copy() for key in saved.files}
            values['variance'] = values['variance'] * factor ** 2
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
        runs[factor] = (run_dir, remote, problem_file, job)
    for factor, (run_dir, remote, _, job) in runs.items():
        if job is None:
            continue
        loop.wait_for_array(args.host, job, [0], args.poll)
        loop.retry(lambda: loop.run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'deployment.tar.gz',
                                     '--exclude', 'inputs.npz', '--exclude', 'study.json', '--exclude',
                                     'prior-precision.npz', f'{args.host}:{remote}/run/', f'{run_dir}/']))
        loop.ssh(args.host, f'rm -rf {remote}/run/member-000/jobs {remote}/run/member-000/checkpoint.npz')

    with open(chain / f'inversion_results_assim_{index}.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    nx, _, nz = problem['dims']
    cells, shape = nx * nz, (nx, nz)
    compare = load_module('compare_runs', 'compare-runs.py')
    boundary = load_module('boundary_check', 'boundary_check.py')
    truth = compare.Truth(json.loads((chain / 'study.json').read_text())['benchmark']).window(saved)['rh']
    kind = boundary.classify(truth)
    dx, _, dz = saved['spacing_ft']
    xo, zo = (np.arange(nx) - nx // 2) * dx, (np.arange(nz) - nz // 2) * dz
    zones = {'near bit': np.outer(np.abs(xo) <= compare.NEAR_X_FT, np.abs(zo) <= compare.ZONE_Z_FT),
             'window': np.ones(shape, bool)}
    ensembles = {'carried prior': (np.asarray(saved['prior_params'])[:cells], None),
                 'chain (no term)': (np.asarray(saved['posterior_ensemble_params']).T[:cells],
                                     np.asarray(saved['posterior_params'][0])[:cells])}
    print('\nMAP fit of the gate point\'s own data, misfit per datum against the measurement covariance')
    for factor, (run_dir, _, problem_file, _) in runs.items():
        if not run_dir.exists() or loop.select_member(run_dir, 1) is None:
            print(f'  std x {factor:g}: not finished')
            continue
        out = laplace_posterior(chain, problem_file, run_dir, args.samples, args.seed)
        ensembles[f'std x {factor:g}'] = (out['models'][:cells], out['model'][:cells])
        np.savez_compressed(run_dir / 'posterior.npz', models=out['models'], map=out['model'])
        print(f'  std x {factor:g}: {out["map_misfit"]:.2f}, {out["result"].get("accepted_iterations")} iterations, '
              f'{out["result"].get("reason")}')
    for zone, inside_zone in zones.items():
        print(f'\n{zone} (log10 Rh, {compare.Truth.__name__.lower()} with layered upscaling)')
        print('  method           class                  cells   error  spread  z-rms  cover90')
        for name, (fields, centre) in ensembles.items():
            field = fields.reshape(nx, nz, -1) / LN10
            std = field.std(axis=2, ddof=1)
            centre = field.mean(axis=2) if centre is None else centre.reshape(shape) / LN10
            inside = (np.quantile(field, .05, axis=2) <= truth) & (truth <= np.quantile(field, .95, axis=2))
            err = centre - truth
            for label, m in [('all cells', inside_zone)] + [(boundary.CLASSES[k], inside_zone & (kind == k))
                                                            for k in range(5)]:
                if m.any():
                    print(f'  {name:16s} {label:21s} {m.sum() / inside_zone.sum():6.1%} '
                          f'{np.sqrt(np.mean(err[m] ** 2)):7.3f} {std[m].mean():7.3f} '
                          f'{np.sqrt(np.mean((err / std)[m] ** 2)):6.2f} {inside[m].mean():8.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
