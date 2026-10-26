"""PET LM-EnRML (approximate update) at one logging point with the v1 2D forward model.

The problem definition (window, tools, 48-channel mask, observations, noise,
stationary Gaussian prior, raw state -> physical map and solver configuration)
is copied from a prepared RML study, so both methods see the identical forward
map: 2 MPI ranks x 8 physical cores, MKL 8 threads, MKL_CBWR pinned.

PET (pipt.update_schemes.enrml.lmenrml_approx) runs unchanged: lambda 'auto'
(0.5 * prior mean data misfit / number of data) and one lambda_factor applied
for both failed (x factor) and successful (/ factor) iterations. Only forward
solves are needed; each forecast is one batch of srun job steps.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from rml_standalone import atomic_json, atomic_npz, factor_apply

ROOT = Path(__file__).resolve().parent
METHOD = 'pet_lmenrml_approx'
STATES = ('lnrh', 'latent')
DATATYPE = 'udar'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_parameters(raw, config):
    """The RML physical map: clip ln Rh, Rv from the latent with the ln Rv cap."""
    import run_DA_sequential as sequential
    cells = len(raw) // 2
    physical = np.asarray(raw, dtype=float).copy()
    physical[:cells] = np.clip(physical[:cells], *config['rh_limits'])
    return sequential._build_param_dict(physical)[0]


def configure_sequential(config):
    import run_DA_sequential as sequential
    sequential.dims = tuple(config['dims'])
    sequential.dX, sequential.dY, sequential.dZ = config['spacing_ft']
    sequential.min_ratio, sequential.max_ratio = config['ratio_min'], config['ratio_max']
    sequential.log_rv_cap = config['log_rv_cap']
    sequential.anisotropy = config.get('anisotropy', 'probit')


def make_solver_forward(config, inputs, output):
    """Forward-only batch of raw states (columns) -> masked predictions, NaN rows for failures."""
    from esmda_remote2d import SimBackendConfig, make_remote2d

    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run solver evaluations within a Slurm allocation')
    configure_sequential(config)
    adapter = make_remote2d(SimBackendConfig(**config['simulator']), tuple(config['dims']),
                            *config['spacing_ft'], config['tools'])
    adapter.jobs_dir = str(output / 'jobs')
    well = {key[5:]: value for key, value in inputs.items() if key.startswith('well_')}
    mask = inputs['mask']

    def forward(raws):
        fields = np.stack([adapter.to_tile(physical_parameters(raw, config)) for raw in raws.T])
        metadata = adapter.build_metadata(config['index'], config['shift_ft'], well,
                                          jacobian=False, tolerate_failures=True)
        predictions = np.asarray(adapter.backend().evaluate_batch(metadata, fields)[0], dtype=float)
        if predictions.shape != (raws.shape[1], 8 * len(config['tools'])):
            raise ValueError(f'Unexpected forward batch shape {predictions.shape}')
        return predictions[:, mask].T

    return forward


class PetForward:
    """PET simulator: one forward batch per forecast; every batch is archived."""

    def __init__(self, forward, archive, concurrency):
        self.forward = forward
        self.archive = Path(archive)
        self.archive.mkdir(parents=True, exist_ok=True)
        self.input_dict = {'parallel': concurrency}
        self.backend = 'pxwf-local-srun'     # enables PET's run_fwd_batch path
        self.redund_sim = None
        self.batches = []

    def run_fwd_batch(self, states, member_indices=None):
        raws = np.column_stack([np.r_[np.ravel(state['lnrh']), np.ravel(state['latent'])]
                                for state in states])
        predictions = self.forward(raws)
        failed = ~np.isfinite(predictions).all(axis=0)
        index = len(self.batches)
        atomic_npz(self.archive / f'forecast-{index:03d}.npz', raw=raws, prediction=predictions,
                   failed=failed)
        self.batches.append((raws, predictions, failed))
        return [False if failed[member] else [{DATATYPE: predictions[:, member].copy()}]
                for member in range(raws.shape[1])]

    def run_fwd_sim(self, state, member_index):
        return self.run_fwd_batch([state], [member_index])[0]

    def predictions_for(self, states):
        """Predictions of the given state columns, found in the archived batches (newest first)."""
        found = np.full((self.batches[0][1].shape[0], states.shape[1]), np.nan)
        for raws, predictions, failed in reversed(self.batches):
            missing = np.flatnonzero(~np.isfinite(found).all(axis=0))
            if not len(missing):
                break
            for member in missing:
                matches = np.flatnonzero((raws == states[:, [member]]).all(axis=0) & ~failed)
                if len(matches):
                    found[:, member] = predictions[:, matches[0]]
        return found


def pet_keys(config, inputs, prior_file, save_folder):
    cells = config['dims'][0] * config['dims'][2]
    stds = config['prior']['stds']
    keys_da = {'daalg': ['enrml', 'lmenrml'], 'analysis': 'approx',
               'iteration': dict(config['pet']),
               'obsname': 'index', 'truedataindex': [config['index']], 'assimindex': [[0]],
               'datatype': [DATATYPE], 'staticvar': list(STATES),
               'truedata': [np.asarray(inputs['observations'], dtype=float)],
               'datavar': ['abs', config.get('data_inflation', 1.) * np.asarray(inputs['variance'], dtype=float)],
               'savefolder': str(save_folder), 'restartsave': 'no'}
    keys_en = {'ne': config['count'], 'state': list(STATES), 'staticvar': list(STATES),
               'importstaticvar': str(prior_file)}
    for name, mean, std in zip(STATES, config['prior']['means'], stds):
        keys_en[f'prior_{name}'] = {'mean': float(mean), 'var': float(std) ** 2,
                                    'grid': [config['dims'][0], config['dims'][2], 1]}
    assert cells == config['dims'][0] * config['dims'][2]
    return keys_da, keys_en


def assimilate(config, inputs, prior_file, forward, output):
    """Run PET's lmenrml_approx unchanged, logging every convergence check."""
    from pipt.loop.assimilation import Assimilate
    from pipt.update_schemes.enrml import lmenrml_approx

    output = Path(output).resolve()
    history = []

    class RecordedLM(lmenrml_approx):
        def calc_prediction(self, enX=None, save_prediction=None):
            # PET bug workaround: for the prior forecast (enX=None) PET sets
            # self.enX = None before its crash-replacement step dereferences it,
            # so one failed prior solve aborts the run. Passing the ensemble
            # explicitly takes PET's own input-ensemble path, which replaces
            # failed members exactly as PET intends and stores the result.
            return super().calc_prediction(enX=self.enX if enX is None else enX,
                                           save_prediction=save_prediction)

        def check_convergence(self):
            lam = self.lam
            conv, success, why = super().check_convergence()
            history.append({'iteration': int(self.iteration), 'lambda_used': float(lam),
                            'lambda_next': float(self.lam), 'success': bool(success),
                            'converged': bool(conv), 'data_misfit': float(self.data_misfit),
                            'data_misfit_std': float(self.data_misfit_std),
                            'candidate_misfit_mean': float(np.mean(self.ensemble_misfit)),
                            'candidate_misfit_std': float(np.std(self.ensemble_misfit)),
                            'prior_data_misfit': float(self.prior_data_misfit)})
            atomic_json(output / 'history.json', history)
            return conv, success, why

    simulator = PetForward(forward, output / 'forecasts', config['resources']['concurrency'])
    keys_da, keys_en = pet_keys(config, inputs, prior_file, output / 'SaveOutputs')
    current = Path.cwd()
    os.chdir(output)            # PET writes its logs and outputs into the working directory
    try:
        np.random.seed(config['data_seed'])   # PET's perturbed observations and crash replacement
        ensemble = RecordedLM(keys_da, keys_en, simulator)
        initial_lambda = ensemble.lam
        Assimilate(ensemble).run()
    finally:
        os.chdir(current)
    final = np.vstack([ensemble.enX[ensemble.idX[name][0]:ensemble.idX[name][1]] for name in STATES])
    prior = np.vstack([ensemble.prior_enX[ensemble.idX[name][0]:ensemble.idX[name][1]] for name in STATES])
    prediction = simulator.predictions_for(final)
    prior_prediction = simulator.batches[0][1]
    atomic_npz(output / 'enrml-result.npz', raw_models=final, predictions=prediction,
               prior_raw_models=prior, prior_predictions=prior_prediction,
               perturbed_observations=ensemble.enObs, observations=inputs['observations'],
               variance=inputs['variance'])
    summary = {'method': METHOD, 'members': int(config['count']), 'initial_lambda_setting': initial_lambda,
               'auto_lambda': history[0]['lambda_used'] if history else None,
               'final_lambda': float(ensemble.lam), 'successful_iterations': int(ensemble.iteration) - 1,
               'forecast_batches': len(simulator.batches),
               'failed_solves': int(sum(failed.sum() for _, _, failed in simulator.batches)),
               'prior_data_misfit': float(ensemble.prior_data_misfit),
               'final_data_misfit': float(ensemble.data_misfit),
               'converged': bool(history and history[-1]['converged']),
               'unmatched_final_predictions': int((~np.isfinite(prediction).all(axis=0)).sum())}
    atomic_json(output / 'summary.json', summary)
    return summary, simulator


def prepare(args):
    import pilot_rml_index45 as runner

    source = args.source.resolve()
    # Reuse the problem definition only; the RML code may have changed since.
    study = json.loads((source / 'study.json').read_text())
    if digest(source / 'inputs.npz') != study['inputs_sha256']:
        raise ValueError('Source study inputs changed')
    with np.load(source / 'inputs.npz', allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in saved.files}
    output = args.run_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a new run directory; prepared prior draws are immutable')
    cascade = None
    data_inflation = 1.
    prior_summary = study['prior']
    if args.selection is not None:
        # Hand-off from a tempered cascade: the best-ranked members start the
        # ensemble, and the data variance carries the remaining likelihood weight.
        from cascade_handoff import load_cascade
        selection = json.loads(args.selection.read_text())
        loaded = load_cascade(selection['cascade'])
        if loaded['provenance']['state_sha256'] != selection['state_sha256']:
            raise ValueError('Cascade state changed since selection')
        if not (np.array_equal(loaded['observations'], inputs['observations'])
                and np.array_equal(loaded['covariance'], inputs['variance'])
                and np.array_equal(loaded['mask'], inputs['mask'])
                and loaded['provenance']['logging_index'] == study['index']):
            raise ValueError('Cascade and source study disagree on the problem definition')
        members = np.asarray(selection['enrml']['members'], dtype=int)
        draws = loaded['fields'][:, members]
        args.count = len(members)
        remaining = 1. - loaded['provenance']['mda_likelihood_weight']
        data_inflation = 1. / remaining
        cells = draws.shape[0] // 2
        prior_summary = {'source': 'selected cascade members',
                         'means': [float(draws[:cells].mean()), float(draws[cells:].mean())],
                         'stds': [float(draws[:cells].std(axis=1, ddof=1).mean()),
                                  float(draws[cells:].std(axis=1, ddof=1).mean())]}
        cascade = {**loaded['provenance'], 'selection': str(args.selection.resolve()),
                   'selection_sha256': digest(args.selection), 'members': members.tolist(),
                   'member_misfits': loaded['misfits'][members].tolist(),
                   'selection_rule': f"best {len(members)} by final cascade-stage misfit",
                   'remaining_likelihood_weight': remaining}
    else:
        prior = runner.load_prior(study)
        draws = prior.mean[:, None] + factor_apply(
            prior, np.random.default_rng(args.seed).normal(size=(len(prior.mean), args.count)))
        cells = len(prior.mean) // 2
    output.mkdir(parents=True)
    atomic_npz(output / 'prior.npz', lnrh=draws[:cells], latent=draws[cells:])
    atomic_npz(output / 'inputs.npz', observations=inputs['observations'],
               variance=inputs['variance'], mask=inputs['mask'],
               **{key: value for key, value in inputs.items() if key.startswith('well_')})
    config = {'method': METHOD, 'index': study['index'], 'count': args.count, 'seed': args.seed,
              'data_seed': args.data_seed, 'dims': study['dims'], 'spacing_ft': study['spacing_ft'],
              'shift_ft': study['shift_ft'], 'tools': study['tools'], 'prior': prior_summary,
              'data_inflation': data_inflation, 'cascade': cascade,
              'ratio_min': study['ratio_min'], 'ratio_max': study['ratio_max'],
              'rh_limits': study['rh_limits'], 'log_rv_cap': study['log_rv_cap'],
              'simulator': {**study['simulator'], 'concurrency': args.concurrency},
              'pet': {'max_iter': args.max_iter, 'lambda': 'auto', 'lambda_factor': args.lambda_factor,
                      'data_misfit_tol': args.data_misfit_tol, 'energy': args.energy},
              'resources': {'concurrency': args.concurrency, 'nodes': args.nodes,
                            'memory_gib_per_node': args.memory_gib, 'walltime': args.walltime},
              'source_study': str(source), 'source_study_sha256': digest(source / 'study.json'),
              'source_inputs_sha256': study['inputs_sha256'],
              'prior_sha256': digest(output / 'prior.npz'),
              'inputs_sha256': digest(output / 'inputs.npz'),
              'code_sha256': {name: digest(ROOT / name) for name in
                              ('enrml_index45.py', 'rml_standalone.py', 'esmda_remote2d.py',
                               'run_DA_sequential.py', 'background_prior.py')}}
    atomic_json(output / 'config.json', config)
    write_sbatch(output, config)
    print(f'Prepared {args.count}-member PET LM-EnRML at index {study["index"]}: {output}')


def write_sbatch(output, config):
    """One driver allocation; each forward solve is an srun step of ranks x cores (physical)."""
    simulator, resources = config['simulator'], config['resources']
    per_node = resources['concurrency'] // resources['nodes']
    if per_node * resources['nodes'] != resources['concurrency']:
        raise ValueError('Concurrent solves must divide evenly over nodes')
    script = f'''#!/bin/bash
#SBATCH --job-name=enrml-index{config['index']}
#SBATCH --partition=comp
#SBATCH --account=energytech
#SBATCH --qos=normal
#SBATCH --nodes={resources['nodes']}
#SBATCH --ntasks-per-node={per_node * simulator['ranks']}
#SBATCH --cpus-per-task={2 * simulator['cpus']}
#SBATCH --mem={resources['memory_gib_per_node']}G
#SBATCH --time={resources['walltime']}
#SBATCH --output=enrml-%j.log
set -euo pipefail
: "${{RML_PROJECT:?Set RML_PROJECT to the deployed code directory}}"
: "${{ENRML_RUN:?Set ENRML_RUN to the prepared run directory}}"
: "${{RML_PYTHON:?Set RML_PYTHON to the project-local HPC Python}}"
export PYTHONPATH="$RML_PROJECT:$RML_PROJECT/pxwf-src:$RML_PROJECT/pet-src:$RML_PROJECT/geostat-src${{PYTHONPATH:+:$PYTHONPATH}}"
export PXWF_SRC="$RML_PROJECT/pxwf-src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
ulimit -s unlimited 2>/dev/null || true
cd "$RML_PROJECT"
exec "$RML_PYTHON" -B -u enrml_index45.py run --run-dir "$ENRML_RUN" ${{ENRML_ARGS:-}}
'''
    (output / 'enrml.sbatch').write_text(script)


def run(args):
    directory = args.run_dir.resolve()
    config = json.loads((directory / 'config.json').read_text())
    for name, expected in (('prior.npz', config['prior_sha256']), ('inputs.npz', config['inputs_sha256'])):
        if digest(directory / name) != expected:
            raise ValueError(f'{name} changed since preparation')
    if any(digest(ROOT / name) != expected for name, expected in config['code_sha256'].items()):
        raise ValueError('Implementation changed since preparation; use the matching code bundle')
    with np.load(directory / 'inputs.npz', allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in saved.files}
    output = directory / (args.label or 'assimilation')
    output.mkdir(exist_ok=False)
    prior_file = directory / 'prior.npz'
    if args.members:
        # Smoke test: a small leading subset of the prepared prior and fewer iterations.
        with np.load(prior_file) as saved:
            atomic_npz(output / 'prior-subset.npz',
                       **{name: saved[name][:, :args.members] for name in STATES})
        prior_file = output / 'prior-subset.npz'
        config = {**config, 'count': args.members}
    if args.max_iter:
        config = {**config, 'pet': {**config['pet'], 'max_iter': args.max_iter}}
    forward = make_solver_forward(config, inputs, output)
    summary, simulator = assimilate(config, inputs, prior_file, forward, output)
    # Re-simulate the mean of the final raw models (the posterior-mean model check).
    with np.load(output / 'enrml-result.npz') as saved:
        mean_model = saved['raw_models'].mean(axis=1)
    mean_prediction = forward(mean_model[:, None])[:, 0]
    summary['mean_model_misfit'] = float(np.mean((mean_prediction - inputs['observations']) ** 2
                                                 / inputs['variance']))
    atomic_npz(output / 'mean-model.npz', raw=mean_model, prediction=mean_prediction)
    atomic_json(output / 'summary.json', summary)
    print(json.dumps(summary))


def collect(args):
    """Per-member chi2/n against perturbed and original data, clip fractions, RML comparison."""
    output = args.run_dir.resolve() / (args.label or 'assimilation')
    config = json.loads((args.run_dir.resolve() / 'config.json').read_text())
    with np.load(output / 'enrml-result.npz') as saved:
        result = {key: saved[key].copy() for key in saved.files}
    variance = result['variance']
    cells = result['raw_models'].shape[0] // 2

    def chi2(predictions, targets):
        return np.mean((predictions - targets) ** 2 / variance[:, None], axis=0)
    perturbed = chi2(result['predictions'], result['perturbed_observations']) / config.get('data_inflation', 1.)
    original = chi2(result['predictions'], result['observations'][:, None])
    lower, upper = config['rh_limits']
    clipped = np.mean((result['raw_models'][:cells] <= lower) | (result['raw_models'][:cells] >= upper), axis=0)
    report = json.loads((output / 'summary.json').read_text())

    def quantiles(values):
        values = values[np.isfinite(values)]
        return dict(zip(('min', 'p10', 'median', 'p90', 'max'),
                        np.quantile(values, [0, .1, .5, .9, 1]).round(3).tolist()))
    report.update({
        'chi2_per_channel_perturbed': quantiles(perturbed),
        'chi2_per_channel_original': quantiles(original),
        'members_chi2_perturbed_le_10': int(np.sum(perturbed <= 10)),
        'prior_chi2_per_channel_original': quantiles(chi2(result['prior_predictions'],
                                                          result['observations'][:, None])),
        'mean_prediction_misfit': float(np.mean((np.nanmean(result['predictions'], axis=1)
                                                 - result['observations']) ** 2 / variance)),
        'rh_clip_fraction': quantiles(clipped),
        'posterior_raw_std_mean': float(result['raw_models'][:cells].std(axis=1, ddof=1).mean()),
        'prior_raw_std_mean': float(result['prior_raw_models'][:cells].std(axis=1, ddof=1).mean())})
    if args.rml is not None:
        rml = json.loads((args.rml / 'summary.json').read_text())
        report['rml_reference'] = {
            'members': len(rml['members']),
            'randomized_misfit': quantiles(np.array([m['randomized_data_misfit'] for m in rml['members']])),
            'original_misfit': quantiles(np.array([m['original_data_misfit'] for m in rml['members']])),
            'mean_prediction_misfit': rml['mean_prediction_misfit'],
            'mean_model_misfit': rml['mean_model_misfit']}
    atomic_json(output / 'collected.json', report)
    print(json.dumps(report, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--source', type=Path, default=ROOT / 'runs/rml-index45-lm-20261019b')
    prepare_parser.add_argument('--count', type=int, default=250)
    prepare_parser.add_argument('--seed', type=int, default=45250)
    prepare_parser.add_argument('--data-seed', type=int, default=45251)
    prepare_parser.add_argument('--max-iter', type=int, default=30)
    prepare_parser.add_argument('--lambda-factor', type=float, default=10.)
    prepare_parser.add_argument('--data-misfit-tol', type=float, default=.01)
    prepare_parser.add_argument('--energy', type=float, default=.95)
    prepare_parser.add_argument('--concurrency', type=int, default=16,
                                help='Concurrent forward solves (each 2 ranks x 8 physical cores)')
    prepare_parser.add_argument('--nodes', type=int, default=4)
    prepare_parser.add_argument('--memory-gib', type=int, default=64, help='Memory per node')
    prepare_parser.add_argument('--walltime', default='02:00:00')
    prepare_parser.add_argument('--selection', type=Path,
                                help='cascade_handoff.py selection.json: start from the selected cascade '
                                     'members and inflate the data variance by the remaining weight')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--members', type=int, help='Smoke test: first N prepared members')
    run_parser.add_argument('--max-iter', type=int, help='Smoke test: override max_iter')
    collect_parser = sub.add_parser('collect')
    collect_parser.add_argument('--rml', type=Path, help='RML run directory for comparison')
    for command in (prepare_parser, run_parser, collect_parser):
        command.add_argument('--run-dir', type=Path, required=True)
    for command in (run_parser, collect_parser):
        command.add_argument('--label', help='Output subdirectory (default: assimilation)')
    args = parser.parse_args(argv)
    if args.command == 'prepare' and (args.count < 2 or args.max_iter < 1 or args.lambda_factor <= 1
                                      or not 0 < args.energy <= 1 or args.concurrency < 1):
        parser.error('Invalid ensemble or PET settings')
    {'prepare': prepare, 'run': run, 'collect': collect}[args.command](args)


if __name__ == '__main__':
    main()
