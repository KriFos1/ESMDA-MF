"""Which prior keeps the cold IA MAP solvable one logging point further on?

For each candidate prior (a JSON config as for ``ia_autoresearch_eval.py``): a cold MAP at
``--index`` with ``--budget`` optimizer seconds, then the MAP model reindexed to the next point's
window and solved there (forward-batch). Reports the MAP fit, its most conductive cells, and
whether the shifted model solves and how far it is from the next point's data. No clipping
anywhere: the question is whether the prior alone keeps the MAP out of the solver's stall
region (Rh below about 0.1 ohm m).
"""

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

import ia_autoresearch_eval as harness
import pilot_rml_index45 as rml
from esmda_core import LocalWindow
from run_ia_chain import INPUTS
from sequential_matern_1d import reindex_window
import run_DA_sequential as sequential

ROOT = Path(__file__).resolve().parent


def shift_model(model, movement, nx=33, nz=80):
    cells = nx * nz
    fields, rest = model[:2 * cells], model[2 * cells:]
    return np.concatenate([reindex_window(fields[:, None], nx, nz, movement)[:, 0], rest])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, action='append', required=True, help='Candidate prior config (repeatable)')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--budget', type=float, default=600.)
    parser.add_argument('--label', default='prior-solvability')
    parser.add_argument('--host', default='HPC')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    base = ROOT / 'autoresearch' / 'ia-eval' / args.label
    shutil.rmtree(base, ignore_errors=True)
    _, _, well, _, values = harness.read_inputs(INPUTS)
    score_sigma = harness.ia_prepare_data().noise_sigma(values, *harness.SCORE_NOISE)
    window = LocalWindow((33, 5, 80), (5., 5., 5.))
    movement = None
    for point in range(args.index + 2):
        _, movement = window.update(tuple(well[key][point, 0] * sequential.meter_to_feet for key in ('X', 'Y', 'Z')))
    cases, jobs = {}, {}
    for path in args.config:
        config = json.loads(path.read_text())
        name = path.stem
        run_dir = base / name / 'map'
        run_dir.parent.mkdir(parents=True)
        info = harness.build_problem(config, args.index, base / name / 'problem-map.npz')
        harness.prepare_study(config, base / name / 'problem-map.npz', run_dir, args.budget)
        remote = f'{harness.HPC_ROOT}/{args.label}-{name}-map'
        jobs[name] = harness.submit(args.host, run_dir, remote, f'{args.label}-{name}', int(args.budget / 60) + 10)
        cases[name] = {'config': config, 'run_dir': run_dir, 'remote': remote, 'info': info}
    harness.wait(args.host, jobs)
    report = {}
    next_jobs = {}
    for name, case in cases.items():
        run_dir, remote, info = case['run_dir'], case['remote'], case['info']
        harness.retry(lambda: harness.run(['rsync', '-a', '--include=fit.npz', '--include=result.json', '--exclude=*',
                                           f'{args.host}:{remote}/run/member-000/', f'{run_dir}/member-000/']))
        timeouts, crashes = harness.solver_failures(args.host, remote)
        info['score_variance'] = score_sigma[args.index].ravel() ** 2
        score, detail = harness.score(run_dir, info)
        report[name] = {'map_score': score, 'map': detail, 'timeouts': timeouts, 'crashes': crashes,
                        'prior_rh_mean': info['rh_mean'], 'prior_rh_std': info['rh_std']}
        fit = run_dir / 'member-000' / 'fit.npz'
        if not fit.exists():
            continue
        model = np.load(fit)['model']
        rh = model[:33 * 80]
        report[name].update(rh_min_ohmm=float(np.exp(rh.min())), rh_max_ohmm=float(np.exp(rh.max())),
                            cells_below_0p3=int((rh < np.log(.3)).sum()), cells_below_0p1=int((rh < np.log(.1)).sum()))
        # The shifted model at the next point: build that problem, start models = the shifted MAP.
        next_dir = base / name / 'next'
        next_info = harness.build_problem(case['config'], args.index + 1, base / name / 'problem-next.npz')
        harness.prepare_study(case['config'], base / name / 'problem-next.npz', next_dir, 60.)
        for item in ('deployment.tar.gz', 'deployment.json'):
            (next_dir / item).unlink()
        with np.load(next_dir / 'inputs.npz', allow_pickle=False) as saved:
            inputs = {key: saved[key].copy() for key in saved.files}
        shifted = shift_model(model, movement)
        np.savez(next_dir / 'inputs.npz', **{**inputs, 'start_models': np.vstack([shifted, inputs['prior_centers'][0]])})
        study = json.loads((next_dir / 'study.json').read_text())
        study['inputs_sha256'] = rml.digest(next_dir / 'inputs.npz')
        (next_dir / 'study.json').write_text(json.dumps(study, indent=2) + '\n')
        rml.bundle(SimpleNamespace(run_dir=next_dir))
        next_remote = f'{harness.HPC_ROOT}/{args.label}-{name}-next'
        next_jobs[name] = harness.submit(args.host, next_dir, next_remote, f'{args.label}-{name}-n', 10, 'forward-batch')
        case.update(next_dir=next_dir, next_remote=next_remote, next_info=next_info)
    harness.wait(args.host, next_jobs)
    for name in next_jobs:
        case = cases[name]
        harness.retry(lambda: harness.run(['rsync', '-a', f'{args.host}:{case["next_remote"]}/run/forward-batch-000/',
                                           f'{case["next_dir"]}/forward-batch-000/']))
        with np.load(case['next_dir'] / 'forward-batch-000' / 'forward-batch.npz') as saved:
            full = saved['predictions_full']
        obs = case['next_info']['observations_full']
        variance = score_sigma[args.index + 1].ravel() ** 2
        own_variance = case['next_info']['sigma_full'] ** 2
        for label, prediction in zip(('shifted_map', 'prior_centre'), full):
            residual = prediction - obs
            phase = np.arange(len(obs)) % 2 == 0
            residual[phase] = (residual[phase] + 180.) % 360. - 180.
            solvable = bool(np.isfinite(prediction).all())
            report[name][f'next_{label}'] = {'solvable': solvable,
                                             'chi2': float(np.mean(residual ** 2 / variance)) if solvable else None,
                                             'chi2_own_noise': float(np.mean(residual ** 2 / own_variance)) if solvable else None}
    args.output.write_text(json.dumps(report, indent=1, default=float) + '\n')
    for name, row in report.items():
        print(f"{name:14s} prior ln Rh {row['prior_rh_mean']:.2f}+-{row['prior_rh_std']:.2f} | MAP chi2 {row['map_score']:7.2f} "
              f"(own noise {row['map'].get('own_chi2', float('nan')):.2f}; {row['map'].get('iterations')} it, "
              f"failed {row['map'].get('failed_trials')}/{row['map'].get('trials')}) | "
              f"Rh min {row.get('rh_min_ohmm', float('nan')):.3f}, cells <0.3: {row.get('cells_below_0p3')} <0.1: {row.get('cells_below_0p1')} | "
              f"next: shifted MAP {row.get('next_shifted_map')} | prior centre {row.get('next_prior_centre')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
