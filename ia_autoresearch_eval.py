"""Score one IA 2D-inversion configuration: reduced chi2 of the best model found within a time budget.

For each logging index the harness builds a single-point problem from the IA inputs
(no chain carry: the prior is the configured climatological Matern prior, sampled into
the ensemble-precision prior exactly as the chain does at its first point), prepares the
MAP study with single-rank node solves, runs it on HPC under ``--budget`` seconds of
optimizer time, pulls the best model and scores its full 72-channel prediction under a
FIXED noise model (cross-tool 2.5%), whatever noise or channel mask the inversion used.
The last line printed is the mean over the indices; ``autoresearch/ia-last-eval.json``
holds the per-index scores and the solver timeout / crash counts for the guard.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

import pilot_rml_index45 as rml
from apparent_dip import latent_dip
from chain_laplace import export_problem
from esmda_core import LocalWindow
from field_trajectory import along_well_trajectory, read_trajectory
from matern_prior import MaternPrior
from run_chain_laplace_loop import ACTIVE_STATES, log, retry, run, ssh
from run_ia_chain import INPUTS, NOISE_FLOOR, lwd_log_values, read_settings
from write_data import read_las
import run_DA_sequential as sequential

ROOT = Path(__file__).resolve().parent
COMPONENTS = ('USDP', 'USDA', 'UADP', 'UADA', 'UHRP', 'UHRA', 'UHAP', 'UHAA')
SCORE_NOISE = ('cross-tool-signal', .025)         # the fixed scoring noise model
SHORT_UHR = [tool * 8 + component for tool in range(3) for component in (4, 5)]
PENALTY = 1e6                                     # score of an index whose run produced no model
BINARY = '/cluster/home/krfo/ia-dip-build-20261030/solver'
HPC_PYTHON = '/cluster/home/krfo/rml-index45-20261019/.venv/bin/python'
HPC_ROOT = '/cluster/home/krfo/ia-autoresearch'


def ia_prepare_data():
    spec = importlib.util.spec_from_file_location('ia_prepare_data', ROOT.parent / 'IA/prepare_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_inputs(inputs):
    """Trajectory, settings, well arrays, tool labels and (points, tools, channels) readings."""
    trajectory = read_trajectory(inputs / 'trajectory.DAT')
    settings = read_settings(inputs / 'inversion_settings.inp')
    well, _ = along_well_trajectory(trajectory)
    tools = np.loadtxt(inputs / 'tool.inp', delimiter=',', skiprows=1, ndmin=2)
    labels, values = [], []
    for freq, spacing in tools:
        _, datum = read_las(inputs / 'logdata' / f'UDAR_{int(spacing)}ft_{int(freq)}kHz.las')
        labels.append((f'{int(freq)}kHz', f'{spacing:g}ft'))
        values.append(datum)
    return trajectory, settings, well, labels, np.stack(values, axis=1)


def channel_mask(sigma, labels, mask):
    """Retained channels (tools x channels, flattened): noise floor plus the configured exclusions."""
    keep = (sigma >= NOISE_FLOOR).ravel()
    if mask.get('short_uhr'):
        keep[SHORT_UHR] = False
    for name in mask.get('components', []):
        keep[COMPONENTS.index(name)::len(COMPONENTS)] = False
    for tool in mask.get('tools', []):
        position = labels.index(tuple(tool))
        keep[position * len(COMPONENTS):(position + 1) * len(COMPONENTS)] = False
    if not keep.any():
        raise ValueError('Every channel is masked')
    return keep


def build_problem(config, index, output, inputs=INPUTS):
    """Export the single-point 2D problem of ``index`` under the configured prior, noise and mask."""
    prepare = ia_prepare_data()
    trajectory, settings, well, labels, values = read_inputs(inputs)
    cell, dip_bounds = settings['cell_ft'], settings['dip_bounds']
    noise, prior_cfg = config['noise'], config['prior']
    sigma = prepare.noise_sigma(values, noise['rule'], noise['fraction'])[index]
    keep = channel_mask(sigma, labels, config.get('mask', {}))
    observations, variance = values[index].ravel(), sigma.ravel() ** 2
    lwd = lwd_log_values(inputs / 'LWD_resistivity.txt', len(trajectory))
    if prior_cfg.get('mode', 'section-global') == 'section-global':
        rh_mean, rh_std = float(lwd.mean()), float(lwd.std())
    else:
        rh_mean, rh_std = float(lwd[index].mean()), float(np.log(2.))
    rh_mean += float(prior_cfg.get('rh_mean_offset', 0.))
    if prior_cfg.get('rh_std') is not None:
        rh_std = float(prior_cfg['rh_std'])
    nx, nz = 33, 80
    ranges = [list(prior_cfg['rh_ranges_ft']), list(prior_cfg['latent_ranges_ft'])]
    prior = MaternPrior((nx, nz), (cell, cell), ranges,
                        [rh_mean, np.log(prior_cfg['ratio_mean'])], [rh_std, prior_cfg['ratio_std']])
    seeds = np.random.SeedSequence(config.get('seed', 2045)).spawn(index + 1)[index].spawn(2)
    state = prior.sample(int(config.get('ensemble', 250)), seeds[0])
    dip_std = float(latent_dip(prior_cfg['dip_std_degrees'], dip_bounds))
    dip = np.random.default_rng(seeds[1]).normal(0., dip_std, state.shape[1])
    window = LocalWindow((nx, 5, nz), (cell, cell, cell))
    for point in range(index + 1):                 # the window anchors at the first point, as in the chain
        shift, movement = window.update(tuple(well[key][point, 0] * sequential.meter_to_feet
                                              for key in ('X', 'Y', 'Z')))
    probe = {'threshold': 0., 'open': True, 'reverted': True, 'pre_update_mean_prediction_misfit': 0.,
             'post_mean_prediction_misfit': 0., 'post_member_median_misfit': 0.}
    fixed = bool(prior_cfg.get('fix_dip', False))      # zero apparent dip, no dip parameter or FD solves
    export_problem(output, index=index, state=state, observations=observations[keep], variance=variance[keep],
                   mask=keep, shift=shift, movement=movement, dims=(nx, 5, nz), spacing=(cell, cell, cell),
                   tools=labels, anisotropy='log', anisotropy_moments=(np.log(prior_cfg['ratio_mean']),
                   prior_cfg['ratio_std']), ranges=ranges, probe=probe, well=well,
                   prior_type='ensemble_precision', dip_latent=None if fixed else dip, dip_bounds=dip_bounds,
                   fixed_zero_dip=fixed)
    return {'rh_mean': rh_mean, 'rh_std': rh_std, 'retained': int(keep.sum()),
            'observations_full': observations, 'keep': keep}


def prepare_study(config, problem, run_dir, budget):
    solver, optimizer = config['solver'], config.get('optimizer', {})
    settings = [f'{key}={value}' for key, value in optimizer.items()] + [f'max_seconds={budget}']
    rml.prepare_laplace(SimpleNamespace(
        run_dir=run_dir, source=ROOT / 'references/map-source', binary_dir=BINARY, orientation='bedding',
        problem=problem, node_solver=(int(solver['threads']), int(solver['concurrency'])),
        solver_tolerance=solver.get('tolerance'), solver_grid=solver.get('grid'),
        evaluation_timeout=solver.get('timeout'), optimizer_setting=settings, hdfql=None))
    study_path = run_dir / 'study.json'
    study = json.loads(study_path.read_text())
    study['save_full_prediction'] = True
    study_path.write_text(json.dumps(study, indent=2) + '\n')
    rml.bundle(SimpleNamespace(run_dir=run_dir))


def submit(host, run_dir, remote, name, minutes):
    ssh(host, f'mkdir -p {remote}')
    retry(lambda: run(['scp', '-q', run_dir / 'deployment.tar.gz', f'{host}:{remote}/']))
    return ssh(host, f'cd {remote} && tar xzf deployment.tar.gz && rm deployment.tar.gz && '
                     f'export RML_PROJECT={remote}/code RML_RUN={remote}/run RML_PYTHON={HPC_PYTHON} RML_NODE=1 && '
                     f'sbatch --parsable --job-name={name} --array=0-0 --ntasks=1 --cpus-per-task=32 --mem=64G '
                     f'--time=00:{minutes:02d}:00 --export=ALL --chdir=$RML_RUN $RML_RUN/rml-array.sbatch',
               capture=True).strip()


def wait(host, jobs, poll=30.):
    pending = dict(jobs)
    while pending:
        time.sleep(poll)
        text = ssh(host, f"sacct -j {','.join(pending.values())} -n -P -o JobID,State", capture=True)
        states = {}
        for line in text.splitlines():
            fields = line.strip().split('|')
            if len(fields) >= 2 and fields[0].endswith('_0'):
                states[fields[0][:-2]] = fields[1].split()[0]
        for index, job in list(pending.items()):
            state = states.get(job)
            if state is not None and state not in ACTIVE_STATES:
                log(f'index {index}: job {job} {state}')
                del pending[index]


def solver_failures(host, remote):
    """(timeouts, crashes) among the member's solver evaluations, from the job receipts."""
    text = ssh(host, f"cd {remote}/run/member-000/jobs 2>/dev/null || exit 0; "
                     f"cat */results/*.json 2>/dev/null | grep -c TimeoutExpired; "
                     f"cat */results/*.json 2>/dev/null | grep -v TimeoutExpired | grep -c '\"state\": \"FAILED\"'; "
                     f"true",
               capture=True).split()
    return (int(text[0]), int(text[1])) if len(text) == 2 else (0, 0)


def score(run_dir, info):
    """Mean reduced chi2 over all 72 channels under the fixed scoring noise, from the best model."""
    fit = run_dir / 'member-000' / 'fit.npz'
    report = run_dir / 'member-000' / 'result.json'
    result = json.loads(report.read_text()) if report.exists() else {}
    if not fit.exists():
        return PENALTY, {'reason': result.get('reason', 'no fit'), 'error': str(result.get('error', ''))[:200]}
    with np.load(fit) as saved:
        full = saved['prediction_full'] if 'prediction_full' in saved else None
        masked = saved['prediction']
    observations, keep = info['observations_full'], info['keep']
    if full is None or not np.isfinite(full).all():
        prediction = np.full(observations.shape, np.nan)
        prediction[keep] = masked
        note = 'full prediction unavailable: masked channels scored as unfit (penalty 100 each)'
    else:
        prediction, note = np.asarray(full, dtype=float), 'full prediction'
    residual = prediction - observations
    phase = np.arange(len(observations)) % 2 == 0
    residual[phase] = (residual[phase] + 180.) % 360. - 180.
    z2 = residual ** 2 / info['score_variance']
    z2 = np.where(np.isfinite(z2), z2, 100.)
    history = result.get('history') or [{}]
    return float(z2.mean()), {'reason': result.get('reason'), 'iterations': result.get('accepted_iterations'),
                              'seconds': result.get('seconds_this_attempt'), 'own_chi2':
                              2 * history[-1].get('likelihood_loss', np.nan) / max(info['retained'], 1),
                              'failed_trials': sum(1 for t in result.get('trial_history', []) if t.get('failed')),
                              'trials': len(result.get('trial_history', [])), 'note': note}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'autoresearch/ia-config.json')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 3])
    parser.add_argument('--budget', type=float, default=120., help='Optimizer seconds per index')
    parser.add_argument('--label', default=None)
    parser.add_argument('--host', default='HPC')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='Score the finished jobs of --label without preparing or submitting again')
    parser.add_argument('--summary', type=Path, default=ROOT / 'autoresearch/ia-last-eval.json')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if config.get('method', 'map') != 'map':
        raise ValueError(f"Unsupported method {config.get('method')!r}")
    label = args.label or f'ia-ar-{time.strftime("%y%m%d-%H%M%S")}'
    base = ROOT / 'autoresearch' / 'ia-eval' / label
    prepare = ia_prepare_data()
    _, _, _, _, values = read_inputs(INPUTS)
    score_sigma = prepare.noise_sigma(values, *SCORE_NOISE)
    infos, jobs, remotes, run_dirs = {}, {}, {}, {}
    for index in args.indices:
        run_dir = base / f'idx{index:03d}'
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        problem = base / f'laplace-problem-{index:03d}.npz'
        if args.resume:
            problem = base / f'laplace-problem-{index:03d}-rescore.npz'
        infos[index] = build_problem(config, index, problem)
        infos[index]['score_variance'] = score_sigma[index].ravel() ** 2
        if not args.resume:
            prepare_study(config, problem, run_dir, args.budget)
        run_dirs[index] = run_dir
        remotes[index] = f'{HPC_ROOT}/{label}-idx{index:03d}'
        log(f'index {index}: prepared ({infos[index]["retained"]} channels, prior ln Rh '
            f'{infos[index]["rh_mean"]:.3f} +- {infos[index]["rh_std"]:.3f})')
    if args.prepare_only:
        print('prepared')
        return 0
    if not args.resume:
        minutes = int(np.ceil(args.budget / 60.)) + 8
        for index in args.indices:
            jobs[index] = submit(args.host, run_dirs[index], remotes[index], f'{label}-{index}', minutes)
            log(f'index {index}: job {jobs[index]}')
        wait(args.host, jobs)
    summary = {'label': label, 'config': config, 'budget': args.budget, 'indices': {}, 'solver_timeouts': 0,
               'solver_crashes': 0}
    scores = []
    for index in args.indices:
        retry(lambda: run(['rsync', '-a', '--include=fit.npz', '--include=result.json', '--exclude=*',
                           f'{args.host}:{remotes[index]}/run/member-000/', f'{run_dirs[index]}/member-000/']))
        timeouts, crashes = solver_failures(args.host, remotes[index])
        value, detail = score(run_dirs[index], infos[index])
        scores.append(value)
        summary['indices'][str(index)] = {'score': value, 'timeouts': timeouts, 'crashes': crashes, **detail}
        summary['solver_timeouts'] += timeouts
        summary['solver_crashes'] += crashes
        log(f'index {index}: score {value:.3f} ({detail}); timeouts {timeouts}, crashes {crashes}')
        ssh(args.host, f'rm -rf {remotes[index]}/run/member-000/jobs')
    summary['score'] = float(np.mean(scores))
    args.summary.write_text(json.dumps(summary, indent=1, default=float) + '\n')
    print(f'{summary["score"]:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
