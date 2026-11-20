#!/usr/bin/env python3
"""Step 2 of the heavy-tailed prior test: Student-t increment prior at one gated 2D point of a finished chain.

The Student-t prior is solved by reweighting rounds. Each round is one ordinary 2D MAP job on the HPC with
a Gaussian increment prior (increment_prior.py) whose face weights come from the previous round's model;
round 1 takes its weights and its start from the chain's own MAP at the point. The Gaussian anchor is the
carried ensemble's cellwise mean with --anchor-factor times its (floored) cellwise std, so the carried
information enters only weakly and the coupling between cells comes from the increments.

After the last round the Gaussian posterior for the final weights is sampled (same Laplace sampling as the
chain) and scored against the truth by distance to the truth boundaries, next to the carried prior, the
chain's Laplace posterior and, if present, the RML ensemble of rml_point_check.py. The chain is not touched.

python -B -u heavy_tail_2d_point.py --chain runs/fault-mean-g25-s100 --index 115 --label fault-mean
python -B -u heavy_tail_2d_point.py --chain runs/fault-mean-g25-s100 --index 115 --label fault-mean --score-only
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

import run_chain_laplace_loop as loop
from chain_laplace import STD_FLOORS, ensemble_moments, laplace_samples, load_problem
from increment_prior import IncrementPrior, student_penalty, student_weights

ROOT = Path(__file__).resolve().parent
LN10 = np.log(10.)
RH_LIMITS = (-1., float(np.log(40.)))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finished(run_dir):
    report = run_dir / 'member-000' / 'result.json'
    return (report.exists() and json.loads(report.read_text()).get('reason') in loop.TERMINAL_REASONS
            and (run_dir / 'member-000' / 'fit.npz').exists() and (run_dir / 'member-000' / 'laplace.npz').exists())


def solve_round(args, run_dir, remote, prior_file, start):
    """Prepare, deploy and run one MAP job; returns when its results are local."""
    def pull():
        loop.retry(lambda: loop.run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'deployment.tar.gz',
                                     '--exclude', 'checkpoint.npz', '--exclude', 'inputs.npz', '--exclude',
                                     'study.json', '--exclude', 'prior-increment.npz',
                                     f'{args.host}:{remote}/run/', f'{run_dir}/']))

    if not run_dir.exists():
        start_file = run_dir.parent / f'{run_dir.name}-start.npz'
        np.savez(start_file, models=start[None, :], labels=np.array(['previous round']))
        loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'prepare-laplace', '--problem', args.problem,
                  '--source', args.source, '--run-dir', run_dir, '--binary-dir', args.binary_dir,
                  '--orientation', args.orientation, '--increment-prior', prior_file,
                  '--start-models', start_file,
                  *(item for setting in args.optimizer_setting for item in ('--optimizer-setting', setting))])
        loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'bundle', '--run-dir', run_dir])
        loop.ssh(args.host, f'mkdir -p {remote}')
        loop.retry(lambda: loop.run(['scp', '-q', run_dir / 'deployment.tar.gz', f'{args.host}:{remote}/']))
        loop.ssh(args.host, f'cd {remote} && tar xzf deployment.tar.gz && rm deployment.tar.gz')
    else:
        pull()
    for submission in range(3):
        if finished(run_dir):
            break
        job = loop.active_job(args.host, f'{remote}/run')
        if job is None:
            job = loop.ssh(args.host, f'cd {remote} && export RML_PROJECT={remote}/code RML_RUN={remote}/run '
                                      f'RML_PYTHON={args.hpc_python} && sbatch --parsable --export=ALL '
                                      f'--array=0 --chdir=$RML_RUN $RML_RUN/rml-array.sbatch', capture=True).strip()
            loop.log(f'{run_dir.name}: job {job} submitted (submission {submission + 1})')
        loop.wait_for_array(args.host, job, [0], args.poll)
        pull()
    if not finished(run_dir):
        raise RuntimeError(f'{run_dir.name} did not finish; see {remote}/run/member-000')
    loop.ssh(args.host, f'rm -rf {remote}/run/member-000/jobs {remote}/run/member-000/checkpoint.npz')


def read_round(run_dir):
    member = run_dir / 'member-000'
    result = json.loads((member / 'result.json').read_text())
    with np.load(member / 'fit.npz') as saved:
        model, z = saved['model'].copy(), saved['z'].copy()
    with np.load(member / 'laplace.npz') as saved:
        jacobian = saved['jacobian'].copy()
    return model, z, jacobian, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--tag', default='', help='Suffix of the run folder (to keep schedules apart)')
    parser.add_argument('--dof-schedule', type=float, nargs='+', default=None,
                        help='Degrees of freedom used for the weights of round 1, 2, ... (the last value repeats). '
                             'Starting high makes the first rounds Gaussian (convex) and lowers the tails '
                             'gradually to --dof')
    parser.add_argument('--scale-schedule', type=float, nargs='+', default=None,
                        help='Factors on --scales for the weights of round 1, 2, ... (the last value repeats)')
    parser.add_argument('--dof', type=float, default=1.)
    parser.add_argument('--scales', type=float, nargs=4, default=(.1, .03, .05, .015),
                        metavar=('RH_VERTICAL', 'RH_LATERAL', 'LATENT_VERTICAL', 'LATENT_LATERAL'),
                        help='Student-t scales of the increments between 5 ft neighbours (state units)')
    parser.add_argument('--anchor-factor', type=float, default=3.,
                        help='Anchor std = factor x the carried ensemble\'s floored cellwise std')
    parser.add_argument('--penalty-weight', type=float, default=1.,
                        help='Factor on the increment penalty (below one: boundaries cost less relative to '
                             'the data and the anchor)')
    parser.add_argument('--tolerance', type=float, default=.05,
                        help='Stop when no state entry moved more than this in a round')
    parser.add_argument('--samples', type=int, default=250)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--score-only', action='store_true', help='Score the rounds already pulled; submit nothing')
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
    chain = args.chain.resolve()
    args.problem = chain / f'laplace-problem-{args.index:03d}.npz'
    problem = load_problem(args.problem)
    nx, _, nz = problem['dims']
    cells, shape = nx * nz, (nx, nz)
    scales = np.asarray(args.scales).reshape(2, 2)
    means, stds = ensemble_moments(problem['prior_ensemble'], cells, STD_FLOORS)
    mean, anchor_std = np.concatenate(means), args.anchor_factor * np.concatenate(stds)
    with open(chain / f'inversion_results_assim_{args.index}.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    chain_map = np.asarray(saved['posterior_params'][0], dtype=float)
    base = ROOT / 'runs' / 'point-tests' / 'ht' / f'ht-idx{args.index:03d}-{args.label}{args.tag}'

    def scheduled(values, number, default):
        return default if not values else values[min(number, len(values)) - 1]
    steps = max(len(args.dof_schedule or []), len(args.scale_schedule or []))
    base.mkdir(parents=True, exist_ok=True)
    (base / 'settings.json').write_text(json.dumps(
        {'chain': str(chain), 'index': args.index, 'dof': args.dof, 'scales': scales.tolist(),
         'anchor_factor': args.anchor_factor, 'penalty_weight': args.penalty_weight, 'rounds': args.rounds,
         'tolerance': args.tolerance,
         'dof_schedule': args.dof_schedule, 'scale_schedule': args.scale_schedule,
         'rule': 'round k: Gaussian increment prior with Student-t weights at the model of round k-1 '
                 '(round 1: the chain MAP), started from that model'}, indent=2) + '\n')

    def penalty(model):
        return float(student_penalty(model, shape, args.dof, scales, mean, anchor_std, args.penalty_weight))

    model, rounds = chain_map, []
    for number in range(1, args.rounds + 1):
        run_dir = base / f'round-{number}'
        remote = f'{args.hpc_root}/{base.name}-round-{number}'
        if args.score_only and not finished(run_dir):
            break
        prior_file = base / f'prior-round-{number}.npz'
        if not prior_file.exists():
            round_dof = scheduled(args.dof_schedule, number, args.dof)
            round_scales = scales * scheduled(args.scale_schedule, number, 1.)
            IncrementPrior(mean, anchor_std,
                           *student_weights(model, shape, round_dof, round_scales, args.penalty_weight),
                           shape).save(prior_file)
        if not args.score_only:
            solve_round(args, run_dir, remote, prior_file, model)
        new, z, jacobian, result = read_round(run_dir)
        last = result['history'][-1]
        move = float(np.max(np.abs(new - model)))
        rounds.append({'round': number, 'run': run_dir, 'prior': prior_file, 'model': new, 'z': z,
                       'jacobian': jacobian})
        loop.log(f'round {number} (dof {scheduled(args.dof_schedule, number, args.dof):g}, scale x'
                 f'{scheduled(args.scale_schedule, number, 1.):g}): {result.get("reason")}, '
                 f'{result.get("accepted_iterations")} iterations, '
                 f'likelihood loss {last["likelihood_loss"]:.1f}, Student-t penalty {penalty(new):.1f}, '
                 f'objective {last["likelihood_loss"] + penalty(new):.1f}, largest move {move:.3f}, '
                 f'chi2/n {2 * last["likelihood_loss"] / len(problem["observations"]):.2f}')
        model = new
        if move < args.tolerance and number >= steps:
            loop.log(f'converged after round {number}')
            break
    if not rounds:
        loop.log('no finished round to score')
        return 1

    # Gaussian posterior for the final weights (the prior the last round used).
    final = rounds[-1]
    prior = IncrementPrior.load(final['prior'])
    with np.load(final['run'] / 'inputs.npz') as inputs:
        variance = inputs['variance'].copy()
    samples, _, _ = laplace_samples(prior, prior.mean, final['z'], final['jacobian'], variance, args.samples,
                                    args.seed)
    samples[:cells] = np.clip(samples[:cells], *RH_LIMITS)
    np.savez_compressed(base / 'posterior.npz', models=samples, map=final['model'],
                        round_models=np.array([r['model'] for r in rounds]))

    compare = load_module('compare_runs', 'compare-runs.py')
    boundary = load_module('boundary_check', 'boundary_check.py')
    truth = compare.Truth(json.loads((chain / 'study.json').read_text())['benchmark']).window(saved)['rh']
    kind = boundary.classify(truth)
    dx, _, dz = saved['spacing_ft']
    xo, zo = (np.arange(nx) - nx // 2) * dx, (np.arange(nz) - nz // 2) * dz
    zones = {'near bit': np.outer(np.abs(xo) <= compare.NEAR_X_FT, np.abs(zo) <= compare.ZONE_Z_FT),
             'window': np.ones(shape, bool)}
    ensembles = {'carried prior': np.asarray(saved['prior_params'])[:cells],
                 'chain Laplace': np.asarray(saved['posterior_ensemble_params']).T[:cells]}
    rml_dir = ROOT / 'runs' / 'point-tests' / 'rml' / f'rml-idx{args.index:03d}-{args.label}'
    members = [np.load(path)['model'][:cells] for path in sorted(rml_dir.glob('member-*/fit.npz'))]
    if len(members) > 2:
        ensembles['chain RML'] = np.array(members).T
    ensembles['Student-t'] = samples[:cells]
    print('\nError of the MAP by round (log10 Rh): '
          + '; '.join(f'{name} {np.sqrt(np.mean((m[:cells].reshape(shape) / LN10 - truth)[zones["near bit"]] ** 2)):.3f}'
                      for name, m in [('chain MAP', chain_map)] + [(f'round {r["round"]}', r['model']) for r in rounds])
          + '  (near bit)')
    for zone, mask in zones.items():
        print(f'\n{zone} (log10 Rh)')
        print('  method         class                  cells   error  spread  z-rms  cover90')
        for name, fields in ensembles.items():
            field = fields.reshape(nx, nz, -1) / LN10
            centre, std = field.mean(axis=2), field.std(axis=2, ddof=1)
            if name == 'Student-t':
                centre = np.clip(final['model'][:cells], *RH_LIMITS).reshape(shape) / LN10
            inside = (np.quantile(field, .05, axis=2) <= truth) & (truth <= np.quantile(field, .95, axis=2))
            err = centre - truth
            for label, m in [('all cells', mask)] + [(boundary.CLASSES[k], mask & (kind == k)) for k in range(5)]:
                if m.any():
                    print(f'  {name:14s} {label:21s} {m.sum() / mask.sum():6.1%} {np.sqrt(np.mean(err[m] ** 2)):7.3f} '
                          f'{std[m].mean():7.3f} {np.sqrt(np.mean((err / std)[m] ** 2)):6.2f} {inside[m].mean():8.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
