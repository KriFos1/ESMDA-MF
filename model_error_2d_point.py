#!/usr/bin/env python3
"""Is the data error model the lever in 2D? One gated point re-solved with a model-error term of the right size.

The chain's own 2D problem at the point (measurement covariance, same carried prior, same starts) is solved
again with the variance Cd (1 + factor^2), where the factor is sized from the residual of the TRUE model at
this point (truth_forward_2d.py), so the truth has a misfit of about one per datum:

  scaled   the benchmark table (build_model_error.py, calm points) times one number c
  own      each channel's own truth residual at this point (the best a diagonal term can do)

Both need the truth: this is a diagnostic of what a correctly sized term would buy, not a field method.
The Laplace posterior of each variant is scored against the truth by distance to the truth boundaries,
next to the carried prior and the chain's posterior. The chain is not touched.

python -B -u model_error_2d_point.py --chain runs/fault4-control-g25-s100 --index 115
python -B -u model_error_2d_point.py --chain runs/fault4-control-g25-s100 --index 115 --score-only
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import run_chain_laplace_loop as loop
from chain_laplace import laplace_posterior, load_problem
from sequential_matern_1d import model_error_factors

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
    parser.add_argument('--variants', nargs='+', default=['scaled', 'own'], choices=['scaled', 'own'])
    parser.add_argument('--table', type=Path, default=ROOT / 'references/model-error-benchmark3-2d.json')
    parser.add_argument('--residuals', type=Path, default=ROOT / 'compare-output/model-error/channels_2d.csv')
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
                        default=['stagnation_tolerance=0.01', 'stagnation_iterations=1', 'max_rounds=2'])
    args = parser.parse_args(argv)
    chain, index = args.chain.resolve(), args.index
    source = chain / f'laplace-problem-{index:03d}.npz'
    problem = load_problem(source)
    mask, measurement = problem['mask'], problem['variance']
    if json.loads((chain / 'study.json').read_text()).get('model_error', 'none') != 'none':
        raise ValueError('Use a chain without a model-error term: its problem holds the measurement covariance')
    with np.load(ROOT / 'runs' / 'point-tests' / 'truth-fwd' / f'truth-fwd-idx{index:03d}' / 'inputs.npz') as inputs:
        if not (np.array_equal(inputs['mask'], mask) and np.allclose(inputs['observations'], problem['observations'])
                and np.allclose(inputs['variance'], measurement)):
            raise ValueError('The truth forward run used other data, mask or covariance than this problem')
    detail = pd.read_csv(args.residuals)
    residual = detail[detail['index'] == index]['truth_2d'].to_numpy()     # (truth response - data) / std
    if residual.size != mask.sum():
        raise ValueError(f'No truth residuals for index {index} in {args.residuals}')
    table = model_error_factors(args.table, problem['tools'])[1][mask]
    scale = brentq(lambda c: np.mean(residual ** 2 / (1. + (c * table) ** 2)) - 1., 1e-3, 1e4)
    factors = {'scaled': scale * table, 'own': np.abs(residual)}
    print(f'index {index}: truth misfit {np.mean(residual ** 2):.1f} per datum against the measurement covariance; '
          f'table factor x {scale:.2f} brings it to 1')

    runs = {}
    for variant in args.variants:
        factor = factors[variant]
        name = f'me2d-idx{index:03d}-{variant}'
        run_dir, remote = ROOT / 'runs' / 'point-tests' / 'me2d' / name, f'{args.hpc_root}/{name}'
        problem_file = ROOT / 'runs' / 'point-tests' / 'me2d' / f'{name}-problem.npz'
        inflation = np.sqrt(1. + factor ** 2)
        print(f'{variant}: std x {inflation.min():.1f} to {inflation.max():.1f} (median {np.median(inflation):.1f}); '
              f'truth misfit {np.mean(residual ** 2 / inflation ** 2):.2f} per datum')
        if not problem_file.exists():
            with np.load(source) as saved:
                values = {key: saved[key].copy() for key in saved.files}
            values['variance'] = measurement * inflation ** 2
            np.savez_compressed(problem_file, **values)
        runs[variant] = (run_dir, remote, problem_file, None)
        if args.score_only:
            continue
        if not run_dir.exists():
            loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'prepare-laplace', '--problem', problem_file,
                      '--source', args.source, '--run-dir', run_dir, '--binary-dir', args.binary_dir,
                      '--orientation', args.orientation, '--starts', '1', '--start-seed', str(2045),
                      '--include-mean-start',
                      *(item for setting in args.optimizer_setting for item in ('--optimizer-setting', setting))])
            loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'bundle', '--run-dir', run_dir])
            loop.ssh(args.host, f'mkdir -p {remote}')
            loop.retry(lambda: loop.run(['scp', '-q', run_dir / 'deployment.tar.gz', f'{args.host}:{remote}/']))
            loop.ssh(args.host, f'cd {remote} && tar xzf deployment.tar.gz && rm deployment.tar.gz')
        count = json.loads((run_dir / 'study.json').read_text())['count']
        job = None
        if loop.select_member(run_dir, count) is None or not all(
                (run_dir / f'member-{m:03d}' / 'laplace.npz').exists() for m in range(count)):
            job = loop.active_job(args.host, f'{remote}/run')
            if job is None:
                job = loop.ssh(args.host, f'cd {remote} && export RML_PROJECT={remote}/code RML_RUN={remote}/run '
                                          f'RML_PYTHON={args.hpc_python} && sbatch --parsable --export=ALL '
                                          f'--array=0-{count - 1} --chdir=$RML_RUN $RML_RUN/rml-array.sbatch',
                               capture=True).strip()
                loop.log(f'{name}: job {job} submitted')
        runs[variant] = (run_dir, remote, problem_file, job)
    for variant, (run_dir, remote, _, job) in runs.items():
        if job is None:
            continue
        count = json.loads((run_dir / 'study.json').read_text())['count']
        loop.wait_for_array(args.host, job, list(range(count)), args.poll)
        loop.retry(lambda: loop.run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'deployment.tar.gz',
                                     '--exclude', 'inputs.npz', '--exclude', 'study.json', '--exclude',
                                     'prior-precision.npz', f'{args.host}:{remote}/run/', f'{run_dir}/']))

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
    print('\nMAP fit, misfit per datum')
    print('  variant          against stage covariance   against measurement covariance   iterations')
    for variant, (run_dir, _, problem_file, _) in runs.items():
        count = json.loads((run_dir / 'study.json').read_text())['count'] if run_dir.exists() else 0
        if not count or loop.select_member(run_dir, count) is None:
            print(f'  {variant}: not finished')
            continue
        out = laplace_posterior(chain, problem_file, run_dir, args.samples, args.seed)
        ensembles[f'term, {variant}'] = (out['models'][:cells], out['model'][:cells])
        np.savez_compressed(run_dir / 'posterior.npz', models=out['models'], map=out['model'])
        against_measurement = float(np.mean((out['prediction'] - problem['observations']) ** 2 / measurement))
        print(f'  {variant:16s} {out["map_misfit"]:24.2f} {against_measurement:32.1f} '
              f'{out["result"].get("accepted_iterations")!s:>12}')
    for zone, inside_zone in zones.items():
        print(f'\n{zone} (log10 Rh)')
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
