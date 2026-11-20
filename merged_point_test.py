#!/usr/bin/env python3
"""Merged logging points at one gated 2D point: does lateral data coverage correct the boundaries?

The chain's own 2D problem at the point (same carried prior, same starts) is solved again with the data of
earlier logging points added: their tool positions are simulated inside the window of the current point.
Every point enters with its variance divided by its share (Cd / share), the rule for data used in several
updates (the shares of one datum over its uses sum to one). Variants are given as INDEX:SHARE lists:

  --variant 115:0.5 110:0.5              current point and the one 25 ft behind
  --variant 115:0.5 110:0.25 105:0.25    three points

In this single-point test the earlier points were already assimilated by the chain at full weight, so
part of their information is in the carried prior as well. The Laplace posterior of each variant is
scored against the truth by distance to the truth boundaries, next to the carried prior and the chain's
posterior. The chain is not touched.

python -B -u merged_point_test.py --chain runs/fault4-control-g25-s100 --index 115 \
    --variant 115:0.5 110:0.5 --variant 115:0.5 105:0.5 --variant 115:0.5 110:0.25 105:0.25
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
    parser.add_argument('--variant', action='append', nargs='+', required=True, metavar='INDEX:SHARE')
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
    measurement = problem['variance']
    if json.loads((chain / 'study.json').read_text()).get('model_error', 'none') != 'none':
        raise ValueError('Use a chain without a model-error term: its problem holds the measurement covariance')

    def checkpoint(point):
        with open(chain / f'inversion_results_assim_{point}.pkl', 'rb') as stream:
            return pickle.load(stream)

    runs = {}
    for entries in args.variant:
        points = [(int(entry.split(':')[0]), float(entry.split(':')[1])) for entry in entries]
        if points[0][0] != index or not all(0 < share <= 1 for _, share in points):
            raise ValueError('A variant starts with the current point and has shares in (0, 1]')
        variant = '-'.join(f'{point}x{share:g}' for point, share in points)
        name = f'merged-idx{index:03d}-{variant}'
        run_dir, remote = ROOT / 'runs' / 'point-tests' / 'merged' / name, f'{args.hpc_root}/{name}'
        problem_file = ROOT / 'runs' / 'point-tests' / 'merged' / f'{name}-problem.npz'
        if not problem_file.exists():
            with np.load(source) as saved:
                values = {key: saved[key].copy() for key in saved.files}
            parts = [(problem['observations'], measurement / points[0][1], problem['mask'])]
            for point, share in points[1:]:
                earlier = checkpoint(point)
                parts.append((np.asarray(earlier['observations']), np.asarray(earlier['data_covariance']) / share,
                              np.asarray(earlier['observation_mask'], bool)))
            values.update(observations=np.concatenate([part[0] for part in parts]),
                          variance=np.concatenate([part[1] for part in parts]),
                          mask=np.concatenate([part[2] for part in parts]),
                          tool_indices=np.array([point for point, _ in points]))
            np.savez_compressed(problem_file, **values)
        print(f'{variant}: ' + ', '.join(f'point {point} with Cd / {share:g}' for point, share in points))
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
                 'chain (current point only)': (np.asarray(saved['posterior_ensemble_params']).T[:cells],
                                     np.asarray(saved['posterior_params'][0])[:cells])}
    print('\nMAP fit, misfit per datum against the measurement covariance')
    print('  variant                              all merged data   current point only   iterations')
    count_current = int(problem['mask'].sum())
    for variant, (run_dir, _, problem_file, _) in runs.items():
        count = json.loads((run_dir / 'study.json').read_text())['count'] if run_dir.exists() else 0
        if not count or loop.select_member(run_dir, count) is None:
            print(f'  {variant}: not finished')
            continue
        out = laplace_posterior(chain, problem_file, run_dir, args.samples, args.seed)
        ensembles[variant] = (out['models'][:cells], out['model'][:cells])
        np.savez_compressed(run_dir / 'posterior.npz', models=out['models'], map=out['model'])
        merged = load_problem(problem_file)
        residual = (out['prediction'] - merged['observations']) ** 2
        # Back to the measurement covariance: the first block is Cd / share of the current point.
        share_current = float(np.median(measurement / merged['variance'][:count_current]))
        print(f'  {variant:36s} {"":>4s}{out["map_misfit"]:10.2f} (stage) '
              f'{np.mean(residual[:count_current] / measurement):14.2f} '
              f'{out["result"].get("accepted_iterations")!s:>12}   current share {share_current:g}')
    for zone, inside_zone in zones.items():
        print(f'\n{zone} (log10 Rh)')
        print('  method                         class                  cells   error  spread  z-rms  cover90')
        for name, (fields, centre) in ensembles.items():
            field = fields.reshape(nx, nz, -1) / LN10
            std = field.std(axis=2, ddof=1)
            centre = field.mean(axis=2) if centre is None else centre.reshape(shape) / LN10
            inside = (np.quantile(field, .05, axis=2) <= truth) & (truth <= np.quantile(field, .95, axis=2))
            err = centre - truth
            for label, m in [('all cells', inside_zone)] + [(boundary.CLASSES[k], inside_zone & (kind == k))
                                                            for k in range(5)]:
                if m.any():
                    print(f'  {name:30s} {label:21s} {m.sum() / inside_zone.sum():6.1%} '
                          f'{np.sqrt(np.mean(err[m] ** 2)):7.3f} {std[m].mean():7.3f} '
                          f'{np.sqrt(np.mean((err / std)[m] ** 2)):6.2f} {inside[m].mean():8.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
