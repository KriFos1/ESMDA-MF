"""Multi-start MAP test of one exported chain gate point, outside the chain.

Prepares ``pilot_rml_index45.py prepare-laplace`` for an exported
``laplace-problem-<index>.npz`` with the same data and noise as the chain's
MAP, under the chain's carried prior (``--prior carried``) or its stationary
Matern prior (``--prior matern``), optionally weakened (``--prior-scale``), and
several starts (``--start-mode``): ``members`` (distinct members of the carried
pre-update ensemble), ``matern`` (the Matern mean plus Matern draws), ``chain``
(the chain's MAP plus carried members), ``benchmark`` (the benchmark window,
layer-averaged, plus the chain's MAP), ``benchmark-perturbed`` (benchmark, chain
MAP and the benchmark plus half the anomaly of carried members) or ``models``
(explicit ``--start-file``). ``--graded`` replaces the uniform 5-ft rows by a
graded vertical grid with the same row count and thickness, 1-ft rows at the
tool (``graded_rows``; the carried ensemble and the chain MAP are re-averaged
onto it). Deploys to HPC, runs the Slurm array (resubmitting members stopped by
their time budget), pulls the results and compares every start with the chain's
own MAP (the control): objective under this run's prior, its prior and data
parts, chi2/n, Rh RMSE against the benchmark and the bit-column profile.
``--forward-only`` evaluates the start models without optimizing (all 64
channels, ``forward-start``) and reports the per-channel residuals.
Reads the chain and control directories only; everything is written to
``--run-dir``. Restart-safe: a rerun adopts the prepared run and any job in flight.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from chain_laplace import load_problem
from pilot_rml_index45 import TERMINAL_REASONS
from run_chain_laplace_loop import ACTIVE_STATES, PYTHON, ROOT, active_job, log, retry, run, ssh

GRADED_BASE_FT = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem', type=Path, required=True, help='Exported laplace-problem-<index>.npz')
    parser.add_argument('--control', type=Path, required=True, help="The chain's finished MAP run directory")
    parser.add_argument('--run-dir', type=Path, required=True, help='New local run directory')
    parser.add_argument('--prior', choices=['carried', 'matern'], default='carried')
    parser.add_argument('--prior-scale', type=float, default=None,
                        help='Multiply every prior std by this factor (>1 weakens the prior)')
    parser.add_argument('--likelihood-weight', type=float, default=None,
                        help="Replace the problem's likelihood weight (Cd / weight)")
    parser.add_argument('--start-mode', default='members',
                        choices=['members', 'matern', 'chain', 'benchmark', 'benchmark-perturbed', 'models'],
                        help='benchmark: the benchmark window itself plus the chain MAP (T0: the truth\'s own '
                             'data fit is the start entry of its optimizer history)')
    parser.add_argument('--start-file', type=Path, default=None, help='models: npz with models k x n and labels')
    parser.add_argument('--starts', type=int, default=4, help='Carried members or Matern draws')
    parser.add_argument('--start-seed', type=int, default=130)
    parser.add_argument('--graded', action='store_true',
                        help='Graded vertical grid: 1-ft rows at the tool, growing away, same rows and thickness')
    parser.add_argument('--unmask', action='store_true',
                        help='Use every channel above the noise floor (drop the global USDP/USDA sub-cell mask)')
    parser.add_argument('--forward-only', action='store_true',
                        help='Evaluate the start models (all channels) instead of optimizing')
    parser.add_argument('--native-run', type=Path, default=None,
                        help='forward-only: a native-model run directory (outputs/UDAR_*.las) to include')
    parser.add_argument('--source', type=Path, default=ROOT / 'runs/rml-index45-lm-20261019b')
    parser.add_argument('--binary-dir', default='/cluster/home/krfo/adjoint-2d-v1-orient')
    parser.add_argument('--orientation', choices=['gradient', 'bedding'], default='bedding')
    parser.add_argument('--host', default='HPC')
    parser.add_argument('--hpc-root', default='/cluster/home/krfo/rml-index45-20261019')
    parser.add_argument('--hpc-python', default='/cluster/home/krfo/rml-index45-20261019/.venv/bin/python')
    parser.add_argument('--poll', type=float, default=60.)
    parser.add_argument('--max-submissions', type=int, default=3)
    args = parser.parse_args(argv)
    if (args.start_mode == 'models') != (args.start_file is not None):
        parser.error('--start-file goes with --start-mode models')
    if args.unmask and args.graded:
        parser.error('--unmask and --graded are separate tests')
    return args


def array_states(text, job):
    """{element: state} of array ``job`` from ``sacct -n -P -o JobID,State`` output."""
    states = {}
    for line in text.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 2 and fields[0].startswith(f'{job}_') and fields[0][len(job) + 1:].isdigit():
            states[int(fields[0][len(job) + 1:])] = fields[1].split()[0]
    return states


def member_result(run_dir, member):
    report = run_dir / f'member-{member:03d}' / 'result.json'
    return json.loads(report.read_text()) if report.exists() else {}


def finished(run_dir, member, forward_only=False):
    if forward_only:
        return (run_dir / f'member-{member:03d}' / 'forward.npz').exists()
    return (member_result(run_dir, member).get('reason') in TERMINAL_REASONS
            and (run_dir / f'member-{member:03d}' / 'fit.npz').exists())


def carried_members(ensemble_size, count, seed):
    """The member choice of ``prepare-laplace --starts`` (same seed, same members)."""
    return np.sort(np.random.default_rng(seed).choice(ensemble_size, count, replace=False))


# -- graded vertical grid ----------------------------------------------------

def graded_rows(total_rows, cells, tool_row, base=GRADED_BASE_FT):
    """Row counts (in ``base``-ft solver rows) of ``cells`` model rows over ``total_rows``.

    The model row holding ``tool_row`` is one solver row thick; rows grow
    geometrically away from it on both sides (integer counts, non-decreasing),
    each side's cell count proportional to its thickness.
    """
    if not 0 <= tool_row < total_rows or cells < 3 or cells > total_rows:
        raise ValueError('Need 3 <= cells <= rows and the tool row inside the window')
    above, below = tool_row, total_rows - tool_row - 1
    n_above = int(round((cells - 1) * above / (above + below)))
    n_below = cells - 1 - n_above

    def side(rows, count):
        if count == 0:
            if rows:
                raise ValueError('Rows left without cells')
            return []
        if rows < count:
            raise ValueError('More cells than rows on one side')
        low, high = 1., 2.
        while sum(high ** k for k in range(count)) < rows:
            high *= 2
        for _ in range(200):
            ratio = .5 * (low + high)
            if sum(ratio ** k for k in range(count)) < rows:
                low = ratio
            else:
                high = ratio
        edges = np.round(np.cumsum([ratio ** k for k in range(count)]) * rows /
                         sum(ratio ** k for k in range(count))).astype(int)
        edges = np.maximum.accumulate(np.maximum(edges, np.arange(1, count + 1)))
        edges[-1] = rows
        sizes = np.sort(np.diff(np.r_[0, edges]))      # growing away from the tool
        if np.any(sizes < 1):
            raise ValueError('Graded grid produced an empty row')
        return sizes.tolist()

    counts = side(above, n_above)[::-1] + [1] + side(below, n_below)
    assert sum(counts) == total_rows and len(counts) == cells
    return counts


def resample_rows(counts_from, counts_to):
    """Overlap-average matrix (n_to x n_from) between two row partitions of the same column."""
    edges_from, edges_to = np.r_[0, np.cumsum(counts_from)], np.r_[0, np.cumsum(counts_to)]
    weights = np.maximum(0., np.minimum(edges_to[1:, None], edges_from[None, 1:])
                         - np.maximum(edges_to[:-1, None], edges_from[None, :-1]))
    return weights / weights.sum(axis=1, keepdims=True)


def resample_state(state, nx, matrix):
    """Apply a row-resampling matrix to every column of both blocks of raw states (n x m or n)."""
    state = np.asarray(state, dtype=float)
    vector = state.ndim == 1
    blocks = state.reshape(2, nx, matrix.shape[1], -1)
    out = np.einsum('ij,bxjm->bxim', matrix, blocks).reshape(2 * nx * matrix.shape[0], -1)
    return out[:, 0] if vector else out


def graded_problem(args, run_dir):
    """The point's problem on the graded grid (saved next to the run directory) and its row counts."""
    problem = load_problem(args.problem)
    nx, _, nz = problem['dims']
    dz = problem['spacing'][2]
    rows = int(round(nz * dz / GRADED_BASE_FT))
    tool_tvd = float(np.asarray(problem['well_Z']).ravel()[problem['index']]) * 3.28084
    tool_row = int(np.floor((tool_tvd - problem['shift_ft']['z']) / GRADED_BASE_FT))
    counts = graded_rows(rows, nz, tool_row)
    uniform = [int(round(dz / GRADED_BASE_FT))] * nz
    matrix = resample_rows(uniform, counts)
    with np.load(args.problem) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays['prior_ensemble'] = resample_state(arrays['prior_ensemble'], nx, matrix)
    arrays['graded_dz'] = np.asarray(counts)
    path = run_dir.parent / f'{run_dir.name}.problem.npz'
    np.savez(path, **arrays)
    return path, counts, matrix


def unmasked_problem(args, run_dir):
    """The point's problem with every channel above the noise floor (no USDP/USDA sub-cell mask)."""
    import pandas as pd
    from esmda_core import noise_floor_mask
    import run_DA_sequential as sequential

    problem = load_problem(args.problem)
    chain_study = json.loads((args.problem.parent / 'study.json').read_text())
    manifest = json.loads((Path(chain_study['reference']) / 'run.json').read_text())
    data, variances = pd.read_pickle(ROOT / manifest['data']), pd.read_pickle(ROOT / manifest['variance'])
    tools = [tuple(tool) for tool in problem['tools']]
    if tools != [tuple(tool) for tool in data.columns]:
        raise ValueError('Problem tools differ from the data columns')
    observations, covariance = sequential.observation_vectors(data, variances, problem['index'], tools)
    old = np.asarray(problem['mask'], dtype=bool)
    if not (np.allclose(observations[old], problem['observations'])
            and np.allclose(covariance[old], problem['variance'])):
        raise ValueError("The chain's reference data do not reproduce the problem's observations")
    mask = ~noise_floor_mask(observations, covariance)
    with np.load(args.problem) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arrays.update(observations=observations[mask], variance=covariance[mask], mask=mask)
    path = run_dir.parent / f'{run_dir.name}.problem.npz'
    np.savez(path, **arrays)
    return path, int(mask.sum()), int(old.sum())


def benchmark_window(chain_dir, problem, z_edges=None):
    """Benchmark Rh/Rv layer-averaged onto the problem's window (or explicit TVD edges), as the raw state."""
    from transition_mda_study import overlap_weights, reference_truth

    if problem['anisotropy'] != 'log_clipped':
        raise ValueError('The benchmark start supports the log_clipped anisotropy (latent = ln Rv/Rh)')
    chain_study = json.loads((Path(chain_dir) / 'study.json').read_text())
    benchmark = Path(json.loads((Path(chain_study['reference']) / 'run.json').read_text())['benchmark'])
    x, z, _, _ = reference_truth(benchmark)
    nx, _, nz = problem['dims']
    dx, _, dz = problem['spacing']
    shift = problem['shift_ft']
    if z_edges is None:
        z_edges = shift['z'] + np.arange(nz + 1) * dz
    wx = overlap_weights(x, shift['x'] + np.arange(nx + 1) * dx, float(np.median(np.diff(x))))
    wz = overlap_weights(z, np.asarray(z_edges, dtype=float), float(np.median(np.diff(z))))
    import h5py
    with h5py.File(benchmark / 'globalmodel.h5', 'r') as handle:
        middle = handle['formation/Rh'].shape[1] // 2
        rh, rv = (handle[f'formation/{name}'][:, middle, :].T for name in ('Rh', 'Rv'))
    # Layered upscaling: Rh is the harmonic (conductance) mean, Rv the arithmetic (series) mean. An
    # arithmetic Rh across a bed boundary near the tool inflates the benchmark's own misfit ~40x.
    fields = {'Rh': np.maximum(1. / (wx @ (1. / rh) @ wz.T), 1e-3), 'Rv': np.maximum(wx @ rv @ wz.T, 1e-3)}
    log_rh = np.log(fields['Rh'])
    return np.r_[log_rh.ravel(), np.maximum(np.log(fields['Rv']) - log_rh, 0.).ravel()]


def start_arguments(args, run_dir, problem_path, counts=None, matrix=None):
    """prepare-laplace start options; explicit start models are written next to the run directory."""
    if args.start_mode == 'members':
        return ['--starts', str(args.starts), '--start-seed', str(args.start_seed)]
    if args.start_mode == 'models':
        return ['--start-models', args.start_file.resolve()]
    problem = load_problem(problem_path)
    nx, _, nz = problem['dims']
    z_edges = (None if counts is None else
               problem['shift_ft']['z'] + GRADED_BASE_FT * np.r_[0, np.cumsum(counts)])
    if args.start_mode == 'matern':
        from matern_prior import MaternPrior
        if counts is not None:
            raise ValueError('Matern starts are defined on the uniform grid only')
        matern = json.loads((args.problem.parent / 'study.json').read_text())['prior']
        prior = MaternPrior((nx, nz), (problem['spacing'][0], problem['spacing'][2]), problem['ranges'].tolist(),
                            matern['means'], matern['stds'])
        models = np.vstack([prior.mean, prior.sample(args.starts, args.start_seed).T])
        labels = ['Matern mean'] + [f'Matern draw {number}' for number in range(args.starts)]
    else:
        with np.load(args.control / 'member-000' / 'fit.npz') as saved:
            chain_map = saved['model'].copy()
        if matrix is not None:
            chain_map = resample_state(chain_map, nx, matrix)
        bench = benchmark_window(args.problem.parent, problem, z_edges)
        members = carried_members(problem['prior_ensemble'].shape[1], args.starts, args.start_seed)
        if args.start_mode == 'benchmark':
            models = np.vstack([bench, chain_map])
            labels = ['benchmark window', 'chain MAP']
        elif args.start_mode == 'benchmark-perturbed':
            ensemble = problem['prior_ensemble']
            anomalies = ensemble[:, members] - ensemble.mean(axis=1, keepdims=True)
            models = np.vstack([bench, chain_map, (bench[:, None] + .5 * anomalies).T])
            labels = ['benchmark window', 'chain MAP'] + [f'benchmark + 0.5 anomaly of member {m}' for m in members]
        else:
            models = np.vstack([chain_map, problem['prior_ensemble'][:, members].T])
            labels = ['chain MAP'] + [f'member {member}' for member in members]
    path = run_dir.parent / f'{run_dir.name}.starts.npz'
    np.savez(path, models=models, labels=np.array(labels))
    return ['--start-models', path]


def solve(args):
    run_dir = args.run_dir.resolve()
    remote = f'{args.hpc_root}/{run_dir.name}'
    problem_path = args.problem.resolve()
    if args.graded or args.unmask:
        problem_path = run_dir.parent / f'{run_dir.name}.problem.npz'
    if not (run_dir / 'deployment.json').exists():
        if run_dir.exists():
            raise FileExistsError(f'{run_dir} exists but was never bundled; inspect it or use a new directory')
        counts = matrix = None
        if args.unmask:
            problem_path, used, before = unmasked_problem(args, run_dir)
            log(f'unmasked problem: {used} channels (chain mask: {before})')
        if args.graded:
            problem_path, counts, matrix = graded_problem(args, run_dir)
            log(f'graded rows (top to bottom, 1-ft units): {counts}')
        run([PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'prepare-laplace', '--problem', problem_path,
             '--source', args.source, '--run-dir', run_dir, '--binary-dir', args.binary_dir,
             '--orientation', args.orientation, '--prior', args.prior,
             *start_arguments(args, run_dir, problem_path, counts, matrix),
             *([] if args.likelihood_weight is None else ['--likelihood-weight', str(args.likelihood_weight)]),
             *([] if args.prior_scale is None else ['--prior-scale', str(args.prior_scale)])])
        run([PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'bundle', '--run-dir', run_dir])
        ssh(args.host, f'mkdir -p {remote}')
        retry(lambda: run(['scp', '-q', run_dir / 'deployment.tar.gz', f'{args.host}:{remote}/']))
        ssh(args.host, f'cd {remote} && tar xzf deployment.tar.gz && rm deployment.tar.gz')
    count = json.loads((run_dir / 'study.json').read_text())['count']
    action = 'RML_ACTION=forward-start ' if args.forward_only else ''

    def pull():
        retry(lambda: run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'deployment.tar.gz',
                           '--exclude', 'inputs.npz', '--exclude', 'study.json', '--exclude', 'prior-precision.npz',
                           f'{args.host}:{remote}/run/', f'{run_dir}/']))

    pull()
    job = active_job(args.host, f'{remote}/run')
    for submission in range(args.max_submissions):
        pending = [member for member in range(count) if not finished(run_dir, member, args.forward_only)]
        if not pending:
            break
        if job is not None:
            log(f'adopting array job {job} already in flight')
        else:
            job = ssh(args.host, f'cd {remote} && export RML_PROJECT={remote}/code RML_RUN={remote}/run '
                                 f'RML_PYTHON={args.hpc_python} {action}&& sbatch --parsable --export=ALL '
                                 f'--array={",".join(map(str, pending))} --chdir=$RML_RUN '
                                 f'$RML_RUN/rml-array.sbatch', capture=True).strip()
            log(f'array job {job} submitted for members {pending} (submission {submission + 1})')
        while True:
            time.sleep(args.poll)
            states = array_states(ssh(args.host, f'sacct -j {job} -n -P -o JobID,State', capture=True), job)
            # Pending elements appear as one bracketed line; wait until each submitted member reports.
            if set(pending) <= set(states) and all(states[member] not in ACTIVE_STATES for member in pending):
                log(f'job {job}: {states}')
                break
        pull()
        job = None
        if not args.forward_only:
            for member in range(count):
                result = member_result(run_dir, member)
                log(f"member {member}: {result.get('reason')}, {result.get('accepted_iterations')} iterations, "
                    f"{result.get('seconds_this_attempt')} s")
    missing = [member for member in range(count) if not finished(run_dir, member, args.forward_only)]
    if missing:
        log(f'members {missing} did not finish within {args.max_submissions} submissions; reporting the rest')
    else:
        # Solver scratch (tens of GB per member) is not needed once the results are local.
        ssh(args.host, f'rm -rf {remote}/run/member-*/jobs {remote}/run/member-*/checkpoint.npz')
    return run_dir, count, problem_path


# -- reports -------------------------------------------------------------------

def solver_grid(study):
    """(expand, fine meta) of a prepared study: models on the uniform solver grid used for scoring."""
    from pilot_rml_index45 import grid_maps

    expand, _, dims, spacing = grid_maps(study)
    return expand, {'dims': tuple(dims), 'spacing_ft': tuple(spacing), 'shift_ft': study['shift_ft']}


def summarize_fit(directory, problem, truth, grid, scoring):
    """Objective under this run's prior (``scoring``: prior, centre, Cd used; None when the model lives on
    another grid), chi2/n against the original Cd, Rh RMSE and the model on the solver grid."""
    from esmda_core import measurement_misfit
    from rml_standalone import factor_solve
    from transition_mda_study import rh_rmse

    expand, meta = grid
    result = json.loads((directory / 'result.json').read_text())
    with np.load(directory / 'fit.npz') as saved:
        model, prediction = saved['model'].copy(), saved['prediction'].copy()
    row = {'reason': result.get('reason'), 'iterations': result.get('accepted_iterations'),
           'seconds': result.get('seconds_this_attempt')}
    if len(prediction) != len(problem['observations']):
        # A model fitted on another channel set (the chain's masked control): no data rescoring here.
        scoring = None
        row['chi2_per_datum'] = float('nan')
    else:
        row['chi2_per_datum'] = measurement_misfit(prediction, problem['observations'], problem['variance'])
    fine = expand(model)
    row['rh_rmse_log10'] = rh_rmse(fine[:, None], meta, truth)
    first = result['history'][0] if result.get('history') else {}
    if scoring is not None:
        prior, center, variance = scoring
        z = factor_solve(prior, model - center)
        prior_loss = .5 * float(z @ z)
        likelihood_loss = .5 * float(np.sum((prediction - problem['observations']) ** 2 / variance))
        weight = float(np.mean(problem['variance'] / variance))
        row.update(objective=prior_loss + likelihood_loss, prior_loss=prior_loss, likelihood_loss=likelihood_loss,
                   start_objective=first.get('objective'),
                   start_chi2_per_datum=(2 * first['likelihood_loss'] / weight / len(variance)
                                         if 'likelihood_loss' in first else None))
    else:
        row.update(objective=None, prior_loss=None, likelihood_loss=None)
    return row, fine


def report(args, run_dir, count, problem_path):
    from pilot_rml_index45 import load_prior
    from transition_mda_study import overlap_weights, reference_truth

    problem = load_problem(problem_path)
    study = json.loads((run_dir / 'study.json').read_text())
    control_study = json.loads((args.control / 'study.json').read_text())
    with np.load(run_dir / 'inputs.npz') as saved:
        inputs = {key: saved[key].copy() for key in ('prior_centers', 'variance')}
    scoring = (load_prior(study, run_dir), inputs['prior_centers'][0], inputs['variance'])
    chain_study = json.loads((args.problem.parent / 'study.json').read_text())
    manifest = json.loads((Path(chain_study['reference']) / 'run.json').read_text())
    truth = reference_truth(Path(manifest['benchmark']))
    grid, control_grid = solver_grid(study), solver_grid(control_study)
    same_grid = study.get('graded_dz') == control_study.get('graded_dz')
    rows, models = {}, {}
    control = 'chain MAP (control, carried prior' + (')' if same_grid else ', its own uniform grid)')
    rows[control], models[control] = summarize_fit(args.control / 'member-000', problem, truth, control_grid,
                                                   scoring if same_grid else None)
    for member in range(count):
        if finished(run_dir, member):
            label = f"start {member}: {study['starts']['labels'][member]}"
            rows[label], models[label] = summarize_fit(run_dir / f'member-{member:03d}', problem, truth, grid,
                                                       scoring)
    # Profiles and the below-bit summary on the 1-ft (or 5-ft) solver grid of this run.
    nx, _, rows_fine = grid[1]['dims']
    dz_fine = grid[1]['spacing_ft'][2]
    x, z, rh, _ = truth
    shift = problem['shift_ft']
    dx = problem['spacing'][0]
    wx = overlap_weights(x, shift['x'] + np.arange(nx + 1) * dx, float(np.median(np.diff(x))))
    wz = overlap_weights(z, shift['z'] + np.arange(rows_fine + 1) * dz_fine, float(np.median(np.diff(z))))
    true_window = -np.log10(wx @ (1. / rh.T) @ wz.T)        # harmonic Rh
    depth = shift['z'] + (np.arange(rows_fine) + .5) * dz_fine
    bit = nx // 2
    bit_z = float(np.asarray(problem['well_Z']).ravel()[problem['index']]) * 3.28084
    body = (depth > bit_z + 15) & (depth < bit_z + 65)
    profiles = {}
    for label, fine in models.items():
        n_rows = len(fine) // (2 * nx)
        column = np.clip(fine[:nx * n_rows].reshape(nx, n_rows)[bit], *study['rh_limits']) / np.log(10.)
        if n_rows != rows_fine:      # control on its own 5-ft grid: put it on this run's rows
            column = np.repeat(column, rows_fine // n_rows)
        profiles[label] = column
        rows[label]['bit_column_rh_rmse_log10'] = float(np.sqrt(np.mean((column - true_window[bit]) ** 2)))
        rows[label]['below_bit_15_65ft_mean_log10_rh'] = float(column[body].mean())
    truth_row = {'below_bit_15_65ft_mean_log10_rh': float(true_window[bit][body].mean())}
    summary = {'index': int(problem['index']), 'problem': str(problem_path), 'control': str(args.control),
               'prior': study['prior'], 'likelihood_weight': study['likelihood_weight'],
               'graded_dz': study.get('graded_dz'), 'starts': study['starts'], 'benchmark': truth_row,
               'scoring': "objectives under this run's prior and Cd / weight; chi2/n against the original Cd",
               'fits': rows}
    (run_dir / 'multistart-summary.json').write_text(json.dumps(summary, indent=2))
    plot(run_dir, problem, study, profiles, true_window[bit], depth, bit_z, rows)
    for label, row in rows.items():
        objective = ('n/a' if row['objective'] is None else
                     f"{row['objective']:.1f} (prior {row['prior_loss']:.1f}, data {row['likelihood_loss']:.1f})")
        log(f"{label}: J {objective}, chi2/n {row['chi2_per_datum']:.3f}, Rh RMSE {row['rh_rmse_log10']:.3f}, "
            f"below-bit log10 Rh {row['below_bit_15_65ft_mean_log10_rh']:.2f}, {row['reason']}")
    log(f"benchmark below-bit log10 Rh {truth_row['below_bit_15_65ft_mean_log10_rh']:.2f}")


def plot(run_dir, problem, study, profiles, true_column, depth, bit_z, rows):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 8), constrained_layout=True)
    axis.step(true_column, depth, where='mid', color='red', lw=2, label='benchmark (layer average)')
    colors = plt.get_cmap('tab10')
    for number, (label, column) in enumerate(profiles.items()):
        style = dict(color='k', lw=2) if number == 0 else dict(color=colors(number), lw=1.2)
        row = rows[label]
        objective = 'n/a' if row['objective'] is None else f"{row['objective']:.0f}"
        axis.step(column, depth, where='mid', label=f"{label}: J {objective}, chi2/n {row['chi2_per_datum']:.2f}",
                  **style)
    axis.axhline(bit_z, color='0.4', ls=':', lw=1)
    axis.set_xlabel('log10 Rh [ohm m] (MAP, bit column)')
    axis.set_ylabel('TVD [ft]')
    axis.set_ylim(bit_z + 120, bit_z - 120)
    axis.grid(alpha=.3)
    axis.legend(fontsize=6, loc='lower left')
    grid = 'graded grid' if study.get('graded_dz') else 'uniform grid'
    scale = study['prior'].get('scale')
    axis.set_title(f"Point {int(problem['index'])}: MAPs ({grid}, prior x{scale or 1:g}, "
                   f"weight {study['likelihood_weight']:g})", fontsize=9)
    figure.savefig(run_dir / 'multistart-profiles.png', dpi=150)
    plt.close(figure)


def forward_report(args, run_dir, count, problem_path):
    """Per-channel residuals (in sigma) of the evaluated models, including the masked channels."""
    import pandas as pd
    import run_DA_sequential as sequential
    from esmda_remote2d import UDAR_COMPONENTS
    from native_model_check import read_prediction

    problem = load_problem(problem_path)
    study = json.loads((run_dir / 'study.json').read_text())
    chain_study = json.loads((args.problem.parent / 'study.json').read_text())
    manifest = json.loads((Path(chain_study['reference']) / 'run.json').read_text())
    data, variances = pd.read_pickle(ROOT / manifest['data']), pd.read_pickle(ROOT / manifest['variance'])
    tools = list(data.columns)
    observations, covariance = sequential.observation_vectors(data, variances, problem['index'], tools)
    sub_cell = np.zeros(len(observations), dtype=bool)
    sub_cell[sequential.component_positions(len(observations))] = True
    floor = sequential.noise_floor_mask(observations, covariance)
    kept = np.asarray(problem['mask'], dtype=bool)
    predictions = {}
    if args.native_run is not None:
        predictions['native benchmark (exact)'] = read_prediction(args.native_run, tools)
    for member in range(count):
        path = run_dir / f'member-{member:03d}' / 'forward.npz'
        if path.exists():
            with np.load(path) as saved:
                predictions[study['starts']['labels'][member]] = saved['prediction_full'].copy()
    sigma = np.sqrt(covariance)
    table = []
    for channel in range(len(observations)):
        tool, component = divmod(channel, len(UDAR_COMPONENTS))
        row = {'tool': f'{tools[tool][1]} {tools[tool][0]}', 'component': UDAR_COMPONENTS[component],
               'observed': float(observations[channel]), 'sigma': float(sigma[channel]),
               'kept': bool(kept[channel]), 'noise_floor': bool(floor[channel]), 'sub_cell': bool(sub_cell[channel])}
        for label, prediction in predictions.items():
            residual = prediction[channel] - observations[channel]
            if component % 2 == 0:       # phases: nearest branch
                residual = (residual + 180.) % 360. - 180.
            row[label] = float(residual / sigma[channel])
        table.append(row)
    frame = pd.DataFrame(table)
    frame.to_csv(run_dir / 'channel-residuals.csv', index=False)
    summary = {}
    for label in predictions:
        squared = frame[label] ** 2
        summary[label] = {'chi2_kept': float(squared[frame['kept']].mean()),
                          'chi2_masked': float(squared[~frame['kept']].mean()),
                          'chi2_all': float(squared.mean()),
                          'chi2_masked_by_reason': {
                              'noise_floor': float(squared[frame['noise_floor']].mean()) if frame['noise_floor'].any() else None,
                              'sub_cell': float(squared[frame['sub_cell']].mean()) if frame['sub_cell'].any() else None},
                          'chi2_kept_by_component': {c: float(squared[frame['kept'] & (frame['component'] == c)].mean())
                                                     for c in UDAR_COMPONENTS
                                                     if (frame['kept'] & (frame['component'] == c)).any()}}
    (run_dir / 'channel-summary.json').write_text(json.dumps(summary, indent=2))
    for label, entry in summary.items():
        log(f"{label}: chi2/n kept {entry['chi2_kept']:.2f}, masked {entry['chi2_masked']:.2f}, "
            f"all {entry['chi2_all']:.2f}; masked by reason {entry['chi2_masked_by_reason']}")
    return frame, summary


def main(argv=None):
    args = parse_args(argv)
    run_dir, count, problem_path = solve(args)
    if args.forward_only:
        forward_report(args, run_dir, count, problem_path)
    else:
        report(args, run_dir, count, problem_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
