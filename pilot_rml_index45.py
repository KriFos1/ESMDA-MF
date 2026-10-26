"""Prepare, run, check and collect a standalone original-prior RML experiment."""

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

import numpy as np

from background_prior import BackgroundPrior
from rml_standalone import (LM_OPTIMIZER, OPTIMIZER, atomic_json, atomic_npz, factor_apply,
                            objective_gradient, optimize_member, optimize_member_lm,
                            randomized_inputs)

ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / 'runs/benchmark3-full-two-step-esmda-b025-t75-20261015-r2/run.json'
# Stop reasons that end a member without a restart (time_limit and execution
# failures are resumable from the checkpoint).
TERMINAL_REASONS = ('relative_gradient', 'objective_stagnation', 'nonlinearity_limited',
                    'iteration_limit', 'gradient_tolerance', 'stalled', 'no_acceptable_trial')
MODULES = ['pilot_rml_index45.py', 'matern_prior.py', 'rml_standalone.py', 'background_prior.py',
           'lm_laplace.py', 'rolling_prior.py', 'esmda_core.py', 'esmda_remote2d.py',
           'run_DA_sequential.py', 'tv_regularization.py', 'sparse_precision.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ScaledPrior:
    """The same Gaussian prior with every standard deviation multiplied by ``factor`` (covariance x factor^2).

    Keeps the base prior's mean; ``factor`` > 1 weakens the prior relative to the data.
    """

    def __init__(self, base, factor):
        if not factor > 0:
            raise ValueError('The prior scale must be positive')
        self.base, self.factor = base, float(factor)
        self.mean = base.mean
        self.std = np.asarray(base.std) * self.factor
        self.shape = getattr(base, 'shape', None)

    def color(self, values, transpose=False):
        return self.factor * self.base.color(values, transpose=transpose)

    def whiten(self, offsets):
        return self.base.whiten(offsets) / self.factor


def load_prior(study, directory=None):
    """The study's prior, multiplied in standard deviation by ``prior['scale']`` when given."""
    base = load_base_prior(study, directory)
    scale = study['prior'].get('scale')
    return base if scale in (None, 1, 1.) else ScaledPrior(base, scale)


def load_base_prior(study, directory=None):
    """Stationary background prior, or a SpatialPrior fitted to a saved incoming ensemble."""
    prior = study['prior']
    if prior.get('type') == 'matern':
        from matern_prior import from_study_prior
        return from_study_prior(prior)
    if prior.get('type') == 'matern_cellwise':
        from matern_prior import MaternPrior
        if directory is None:
            raise ValueError('A cellwise Matern prior needs the run directory')
        with np.load(Path(directory) / prior['file']) as saved:
            means, stds = saved['means'].copy(), saved['stds'].copy()
        return MaternPrior(tuple(prior['shape']), tuple(prior['spacing']), prior['lengths'],
                           list(means), list(stds), robin=prior.get('boundary', 'robin') == 'robin')
    if prior.get('type') == 'ensemble_precision':
        from sparse_precision import EnsemblePrecisionPrior
        if directory is None:
            raise ValueError('An ensemble-precision prior needs the run directory')
        return EnsemblePrecisionPrior.load(Path(directory) / prior['file'])
    if prior.get('type') == 'ensemble_spatial':
        from lm_laplace import SpatialPrior
        if directory is None:
            raise ValueError('An ensemble-based prior needs the run directory')
        with np.load(Path(directory) / prior['file']) as saved:
            ensemble = saved['fields'].copy()
        return SpatialPrior(ensemble, tuple(prior['shape']), tuple(prior['spacing']),
                            [list(block) for block in prior['lengths']], tuple(prior['std_floor']))
    return BackgroundPrior(**prior)


def prepare(args):
    import h5py
    import pandas as pd
    from esmda_core import LocalWindow, noise_floor_mask, observation_vectors
    from esmda_remote2d import SimBackendConfig
    import run_DA_sequential as sequential

    manifest = json.loads(args.reference.read_text())
    if manifest['prior_style'] != 'gaussian' or manifest['prior_projection'] != 'off':
        raise ValueError('Require original Gaussian prior without mode projection')
    output = args.run_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a new run directory; prepared randomizations are immutable')
    nx, ny, nz = manifest['dims']
    spacing = manifest['spacing_ft']
    with h5py.File(Path(manifest['benchmark']) / 'globalmodel.h5') as handle:
        well = {key: value[:] for key, value in handle['wellpath'].items()}
    if not 0 <= args.index < len(well['X']):
        raise ValueError('Logging index outside wellpath')
    window = LocalWindow((nx, ny, nz), tuple(spacing))
    for index in range(args.index + 1):
        shift, _ = window.update(tuple(float(well[key][index, 0]) * sequential.meter_to_feet
                                       for key in ('X', 'Y', 'Z')))
    data = pd.read_pickle(ROOT / manifest['data'])
    variances = pd.read_pickle(ROOT / manifest['variance'])
    tools = list(data.columns)
    observations, variance = observation_vectors(data, variances, args.index, tools)
    shallow = np.zeros(len(observations), dtype=bool)
    shallow[sequential.component_positions(len(observations))] = True
    mask = ~(noise_floor_mask(observations, variance) | shallow)
    if not mask.any():
        raise ValueError('No retained observation channels')
    fraction = manifest['prior_reservoir_fraction']
    background, reservoir, mode_std = [manifest[key] * np.log(10.) for key in
                                      ('prior_background', 'prior_reservoir', 'prior_mode_std')]
    prior = {'shape': [nx, nz], 'spacing': [spacing[0], spacing[2]],
             'lengths': [[50., 50.], [manifest['prior_latent_correlation_x_ft'],
                                     manifest['prior_latent_correlation_z_ft']]],
             'means': [fraction * reservoir + (1 - fraction) * background, 0.],
             'stds': [float(np.sqrt(mode_std ** 2 + fraction * (1 - fraction)
                                    * (reservoir - background) ** 2)),
                      manifest['rv_parameterization']['latent_std']]}
    # Each member's allocation holds `candidates` concurrent srun solver steps of
    # `ranks` MPI ranks x `cpus` physical cores with MKL-threaded DGEMM.
    config = SimBackendConfig(backend='local', cpus=args.cpus, ranks=args.ranks,
                              concurrency=args.candidates, workers=1,
                              binary_dir=args.binary_dir, oneapi=args.oneapi, hdfql=args.hdfql,
                              solver_tolerance=args.solver_tolerance,
                              mkl_threads=args.mkl_threads, mkl_cbwr=args.mkl_cbwr,
                              srun_steps=True, physical_cores=True)
    study = {'method': 'standalone_original_prior_rml', 'index': args.index,
             'count': args.count, 'seed': args.seed, 'dims': [nx, ny, nz],
             'spacing_ft': spacing, 'shift_ft': shift, 'tools': tools, 'prior': prior,
             'ratio_min': 1., 'ratio_max': manifest['prior_ratio_max'],
             'rh_limits': [-1., float(np.log(40.))], 'log_rv_cap': float(np.log(40.)),
             'parameterization': 'raw Gaussian state; Rh clipped only in physical forward map',
             'covariance_note': 'squared-exponential Rh 50/50 ft; exponential latent; axis nugget 1e-10',
             'source_manifest_sha256': digest(args.reference),
             'source_sha256': {name: digest(ROOT / name) for name in MODULES},
             'optimizer_method': LM_OPTIMIZER,
             'optimizer': {'candidates': args.candidates, 'initial_trust': args.initial_trust,
                           'max_trust': args.max_trust, 'min_trust': args.min_trust,
                           'max_iterations': args.max_iterations, 'max_rounds': args.max_rounds,
                           'max_seconds': args.max_seconds,
                           'stagnation_tolerance': args.stagnation_tolerance,
                           'stagnation_iterations': args.stagnation_iterations,
                           'relative_gradient_tolerance': args.relative_gradient_tolerance},
             'resources': {'memory_gib': args.memory_gib, 'walltime': args.walltime},
             'simulator': asdict(config)}
    inputs = randomized_inputs(load_prior(study), observations[mask], variance[mask],
                               args.count, args.seed)
    input_values = {**inputs, 'mask': mask,
                    **{f'well_{key}': value for key, value in well.items()}}
    reused_bytes = None
    if args.reuse_inputs is not None:
        source = args.reuse_inputs / 'inputs.npz'
        with np.load(source, allow_pickle=False) as saved:
            if set(saved.files) != set(input_values) or any(
                    not np.array_equal(saved[key], value) for key, value in input_values.items()):
                raise ValueError('Requested reused inputs differ from this statistical experiment')
        reused_bytes = source.read_bytes()
        study['reused_inputs_sha256'] = digest(source)
    output.mkdir(parents=True)
    if reused_bytes is None:
        atomic_npz(output / 'inputs.npz', **input_values)
    else:
        (output / 'inputs.npz').write_bytes(reused_bytes)
    study['inputs_sha256'] = digest(output / 'inputs.npz')
    atomic_json(output / 'study.json', study)
    write_array(output, study)
    print(f'Prepared {args.count} fixed RML problems at index {args.index}: {output}')


def prepare_cascade(args):
    """RML for the remaining likelihood after a tempered 0D/1D cascade.

    Starts (= prior centres) are cascade members chosen by cascade_handoff.
    The prior covariance is a SpatialPrior fitted to the full cascade ensemble,
    as in the cascade's own 2D stage. Data variance is inflated by the remaining
    likelihood weight (alpha = 1 / remaining), and targets are perturbed with it.
    """
    from cascade_handoff import load_cascade

    output = args.run_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a new run directory; prepared randomizations are immutable')
    source = json.loads((args.source / 'study.json').read_text())
    with np.load(args.source / 'inputs.npz', allow_pickle=False) as saved:
        source_inputs = {key: saved[key].copy() for key in saved.files}
    selection = json.loads(args.selection.read_text())
    cascade = load_cascade(selection['cascade'])
    if cascade['provenance']['state_sha256'] != selection['state_sha256']:
        raise ValueError('Cascade state changed since selection')
    if not (np.array_equal(cascade['observations'], source_inputs['observations'])
            and np.allclose(cascade['covariance'], source_inputs['variance'], rtol=0, atol=0)
            and np.array_equal(cascade['mask'], source_inputs['mask'])):
        raise ValueError('Cascade and source study disagree on observations, noise or channel mask')
    if cascade['provenance']['logging_index'] != source['index']:
        raise ValueError('Cascade and source study are at different logging points')
    members = np.asarray(selection['rml']['members'], dtype=int)
    if args.prior == 'original':
        return prepare_cascade_starts(args, output, source, source_inputs, selection, cascade, members)
    if args.prior == 'cascade-prior':
        return prepare_cascade_prior_starts(args, output, source, source_inputs, selection, cascade,
                                            members)
    remaining = 1. - cascade['provenance']['mda_likelihood_weight']
    inflation = 1. / remaining
    nx, _, nz = source['dims']
    variance = source_inputs['variance'] * inflation
    noise = (np.random.default_rng(args.noise_seed).normal(size=(len(members), len(variance)))
             * np.sqrt(variance))
    output.mkdir(parents=True)
    atomic_npz(output / 'prior-ensemble.npz', fields=cascade['fields'])
    atomic_npz(output / 'inputs.npz', prior_centers=cascade['fields'][:, members].T,
               measurement_noise=noise, targets=source_inputs['observations'][None, :] + noise,
               observations=source_inputs['observations'], variance=variance,
               original_variance=source_inputs['variance'], mask=source_inputs['mask'],
               cascade_members=members,
               **{key: value for key, value in source_inputs.items() if key.startswith('well_')})
    study = {**source, 'method': 'cascade_tempered_rml', 'count': len(members),
             'seed': args.noise_seed,
             'prior': {'type': 'ensemble_spatial', 'file': 'prior-ensemble.npz', 'shape': [nx, nz],
                       'spacing': [source['spacing_ft'][0], source['spacing_ft'][2]],
                       'lengths': source['prior']['lengths'], 'std_floor': [.23, .1]},
             'covariance_note': ('SpatialPrior of the full cascade ensemble: cellwise std (floors '
                                 '.23/.1), separable exponential correlation with the original lengths'),
             'cascade': {**cascade['provenance'], 'selection': str(args.selection.resolve()),
                         'selection_sha256': digest(args.selection), 'members': members.tolist(),
                         'member_misfits': cascade['misfits'][members].tolist(),
                         'selection_rule': (f"random {selection['rml']['count']} from the best "
                                            f"{selection['rml']['quantile']:.0%} by final misfit")},
             'data_inflation': inflation, 'remaining_likelihood_weight': remaining,
             'source_study': str(args.source.resolve()),
             'source_sha256': {name: digest(ROOT / name) for name in MODULES}}
    study.pop('reused_inputs_sha256', None)
    study['inputs_sha256'] = digest(output / 'inputs.npz')
    atomic_json(output / 'study.json', study)
    write_array(output, study)
    print(f'Prepared {len(members)} cascade-started RML problems (data inflation {inflation:g}): {output}')


def prepare_cascade_starts(args, output, source, source_inputs, selection, cascade, members):
    """Cascade members only as optimizer starts: original prior, untempered data.

    The first len(members) randomized problems of the source study (prior centres
    drawn from the original stationary prior, targets d + N(0, Cd)) are reused
    unchanged, so optimizer j minimizes exactly the source study's objective j;
    only its starting model is the j-th selected cascade member.
    """
    count = len(members)
    if count > source_inputs['prior_centers'].shape[0]:
        raise ValueError('Source study has fewer randomized problems than cascade starts')
    output.mkdir(parents=True)
    atomic_npz(output / 'inputs.npz',
               **{key: source_inputs[key][:count] for key in ('prior_centers', 'measurement_noise', 'targets')},
               start_models=cascade['fields'][:, members].T, cascade_members=members,
               observations=source_inputs['observations'], variance=source_inputs['variance'],
               mask=source_inputs['mask'],
               **{key: value for key, value in source_inputs.items() if key.startswith('well_')})
    study = {**source, 'method': 'cascade_started_rml_original_prior', 'count': count,
             'cascade': {**cascade['provenance'], 'selection': str(args.selection.resolve()),
                         'selection_sha256': digest(args.selection), 'members': members.tolist(),
                         'member_misfits': cascade['misfits'][members].tolist(),
                         'selection_rule': (f"random {selection['rml']['count']} from the best "
                                            f"{selection['rml']['quantile']:.0%} by final misfit"),
                         'role': 'optimizer starting models only'},
             'data_inflation': 1.,
             'randomization_note': (f'problems 0..{count - 1} of the source study, unchanged '
                                    '(original prior centres, d + N(0, Cd) targets)'),
             'source_study': str(args.source.resolve()),
             'source_inputs_sha256': source['inputs_sha256'],
             'source_sha256': {name: digest(ROOT / name) for name in MODULES}}
    study.pop('reused_inputs_sha256', None)
    study['inputs_sha256'] = digest(output / 'inputs.npz')
    atomic_json(output / 'study.json', study)
    write_array(output, study)
    print(f'Prepared {count} cascade-started RML problems with the original prior: {output}')


def cascade_matern_prior(cascade_study):
    """Study prior dictionary reproducing the Matern prior a cascade drew its ensemble from."""
    import pilot_cascade_laplace_index34 as pilot

    if cascade_study.get('prior_family') != 'matern':
        raise ValueError('The cascade did not draw from a Matern prior')
    manifest = json.loads((Path(cascade_study['reference']) / 'run.json').read_text())
    anisotropy = cascade_study['anisotropy']['parameterization']
    latent_mean, latent_std = cascade_study['anisotropy']['state_moments']
    rh_mean, rh_std = pilot.rh_prior_moments(manifest)
    prior = {'type': 'matern', 'shape': [manifest['dims'][0], manifest['dims'][2]],
             'spacing': [manifest['dx_ft'], manifest['dz_ft']],
             'lengths': [[pilot.RH_CORRELATION_X_FT, pilot.RH_CORRELATION_Z_FT],
                         [manifest['prior_latent_correlation_x_ft'],
                          manifest['prior_latent_correlation_z_ft']]],
             'means': [float(rh_mean), float(latent_mean)], 'stds': [float(rh_std), float(latent_std)],
             'boundary': 'robin'}
    return prior, anisotropy, manifest


def prepare_cascade_prior_starts(args, output, source, source_inputs, selection, cascade, members):
    """Cascade members only as optimizer starts, under the cascade's own (Matern) prior.

    Nothing from the cascade enters the RML objective: prior centres are fresh
    independent draws from the prior the cascade started from, targets are
    d + N(0, Cd) with the untempered Cd. The selected cascade members are used
    only as starting models.
    """
    from matern_prior import from_study_prior
    from pilot_cascade_laplace_index34 import cap_rh, clamp_log_rh, matern_reference
    from rml_standalone import randomized_inputs

    cascade_study = json.loads((Path(cascade['provenance']['cascade']) / 'study.json').read_text())
    if cascade_study['mda_likelihood_weight'] != 1.:
        raise ValueError('Starts-only hand-off expects a full-energy cascade')
    prior, anisotropy, manifest = cascade_matern_prior(cascade_study)
    latent_mean, latent_std = cascade_study['anisotropy']['state_moments']
    scorer = from_study_prior(prior)
    reference = matern_reference(manifest, anisotropy, latent_mean, latent_std)
    # The study prior must be the cascade's prior: same draws, and it reproduces
    # the cascade's saved prior ensemble (clamped and capped as in the pilot).
    with np.load(Path(cascade['provenance']['cascade']) / 'state-00-prior.npz') as saved:
        cascade_prior = saved['fields'].copy()
    redrawn = cap_rh(clamp_log_rh(scorer.sample(cascade_prior.shape[1], cascade_study['seed'])), 40.)
    if not (np.array_equal(scorer.sample(2, 0), reference.sample(2, 0))
            and np.allclose(redrawn, cascade_prior, rtol=0, atol=1e-12)):
        raise ValueError('Study prior does not reproduce the cascade prior')
    count = len(members)
    inputs = randomized_inputs(scorer, source_inputs['observations'], source_inputs['variance'],
                               count, args.noise_seed)
    output.mkdir(parents=True)
    atomic_npz(output / 'inputs.npz', **inputs, start_models=cascade['fields'][:, members].T,
               cascade_members=members, mask=source_inputs['mask'],
               **{key: value for key, value in source_inputs.items() if key.startswith('well_')})
    study = {**source, 'method': 'cascade_started_rml_cascade_prior', 'count': count,
             'seed': args.noise_seed, 'prior': prior, 'anisotropy': anisotropy,
             'anisotropy_state_moments': [latent_mean, latent_std],
             'parameterization': ('raw Gaussian state; Rh clipped only in physical forward map; '
                                  f'{cascade_study["anisotropy"]["map"]}'),
             'covariance_note': ('Matern nu=1 SPDE prior of the cascade (sparse precision, Robin '
                                 'boundary, cellwise std); lengths are practical ranges'),
             'cascade': {**cascade['provenance'], 'selection': str(args.selection.resolve()),
                         'selection_sha256': digest(args.selection), 'members': members.tolist(),
                         'member_misfits': cascade['misfits'][members].tolist(),
                         'selection_rule': (f"random {selection['rml']['count']} from the best "
                                            f"{selection['rml']['quantile']:.0%} by final misfit"),
                         'schedule': cascade_study['schedule'],
                         'role': 'optimizer starting models only'},
             'data_inflation': 1.,
             'randomization_note': ('fresh independent prior centres from the cascade prior and '
                                    'd + N(0, Cd) targets (seed = noise seed)'),
             'source_study': str(args.source.resolve()),
             'source_sha256': {name: digest(ROOT / name) for name in MODULES}}
    study.pop('reused_inputs_sha256', None)
    study['inputs_sha256'] = digest(output / 'inputs.npz')
    atomic_json(output / 'study.json', study)
    write_array(output, study)
    print(f'Prepared {count} cascade-started RML problems with the cascade (Matern) prior: {output}')


def prepare_laplace(args):
    """One MAP problem for a chain gate point (chain_laplace.export_problem).

    Prior (the problem's ``prior_type``): Matern with the carried ensemble's
    cellwise mean and floored std and the chain's ranges, or the ensemble's own
    sparse precision (local ridge modified Cholesky, ``sparse_precision``).
    Data: the point's observations, unperturbed, with Cd / likelihood_weight
    (the share of the data energy left for 2D). Start: the prior mean, or with
    ``--starts N`` N members sharing that prior and data that start from N
    distinct members of the carried ensemble (``--start-seed``), or one member
    per model of ``--start-models`` (npz: ``models`` k x n, ``labels``): a
    multi-start test of the same MAP problem. ``--prior matern`` replaces the
    carried prior by the chain's own stationary Matern prior (its means, stds and
    ranges, no memory of earlier points). ``--likelihood-weight w`` replaces the
    problem's weight (data covariance Cd / w). The member job also saves the Jacobian
    at the MAP for the Laplace resampling (``save_jacobian``).
    """
    from chain_laplace import PRECISION_RADIUS, STD_FLOORS, ensemble_moments, load_problem
    from sparse_precision import EnsemblePrecisionPrior

    output = args.run_dir.resolve()
    if output.exists():
        raise FileExistsError('Use a new run directory; prepared problems are immutable')
    source = json.loads((args.source / 'study.json').read_text())
    simulator = {key: value for key, value in (('binary_dir', args.binary_dir),
                                               ('orientation', args.orientation)) if value is not None}
    source['simulator'] = {**source['simulator'], **simulator}
    for setting in getattr(args, 'optimizer_setting', None) or []:
        key, separator, value = setting.partition('=')
        if not separator or key not in source.get('optimizer', {}):
            raise ValueError(f'Unknown optimizer setting {setting!r}; known: {sorted(source.get("optimizer", {}))}')
        current = source['optimizer'][key]
        source['optimizer'] = {**source['optimizer'], key: type(current)(float(value) if isinstance(current, float)
                                                                         else int(value))}
    node = getattr(args, 'node_solver', None)
    if node is not None:
        threads, concurrency = (int(value) for value in node)
        if threads < 1 or concurrency < 1:
            raise ValueError('Node solver threads and concurrency must be positive')
        # Single-rank standalone solves (no srun); fwd2D handles both receiver groups in one process.
        source['simulator'] = {**source['simulator'], 'ranks': 1, 'cpus': threads, 'mkl_threads': threads,
                               'concurrency': concurrency, 'srun_steps': False, 'physical_cores': False,
                               **({'hdfql': str(args.hdfql)} if getattr(args, 'hdfql', None) else {})}
        source['execution'], source['clean_jobs'] = 'node', True
    problem_path = args.problem.resolve()
    problem = load_problem(problem_path)
    nx, ny, nz = problem['dims']
    dx, dy, dz = problem['spacing']
    weight = problem['likelihood_weight']
    if getattr(args, 'likelihood_weight', None) is not None:
        if not args.likelihood_weight > 0:
            raise ValueError('The likelihood weight must be positive')
        weight = float(args.likelihood_weight)
    observations, variance = problem['observations'], problem['variance']
    output.mkdir(parents=True)
    if getattr(args, 'prior', 'carried') == 'matern':
        matern = json.loads((problem_path.parent / 'study.json').read_text())['prior']
        if matern.get('family') != 'matern':
            raise ValueError('The chain study does not record a Matern prior')
        means = [np.full(nx * nz, float(value)) for value in matern['means']]
        stds = [np.full(nx * nz, float(value)) for value in matern['stds']]
        atomic_npz(output / 'prior-moments.npz', means=np.vstack(means), stds=np.vstack(stds))
        center = np.concatenate(means)
        prior_spec = {'type': 'matern_cellwise', 'file': 'prior-moments.npz', 'shape': [nx, nz],
                      'spacing': [dx, dz], 'lengths': problem['ranges'].tolist(), 'boundary': 'robin',
                      'stationary': {'means': matern['means'], 'stds': matern['stds']}}
        covariance_note = (f"the chain's stationary Matern nu=1 prior (means {matern['means']}, stds "
                           f"{matern['stds']}, ranges {problem['ranges'].tolist()}); no carried information")
    elif problem['prior_type'] == 'ensemble_precision':
        prior = EnsemblePrecisionPrior.fit(problem['prior_ensemble'], (nx, nz), PRECISION_RADIUS, STD_FLOORS)
        prior.save(output / 'prior-precision.npz')
        center = prior.mean
        prior_spec = {'type': 'ensemble_precision', 'file': 'prior-precision.npz', 'shape': [nx, nz],
                      'radius': PRECISION_RADIUS, 'std_floors': list(STD_FLOORS),
                      'sha256': digest(output / 'prior-precision.npz'), 'fit': prior.summary()}
        covariance_note = ('the carried ensemble\'s mean and sparse precision: ridge regressions (GCV) of each '
                           f'variable on its predecessors within {PRECISION_RADIUS} cells, both blocks; '
                           f'marginal std floors {STD_FLOORS}')
    else:
        means, stds = ensemble_moments(problem['prior_ensemble'], nx * nz, STD_FLOORS)
        atomic_npz(output / 'prior-moments.npz', means=np.vstack(means), stds=np.vstack(stds))
        center = np.concatenate(means)
        prior_spec = {'type': 'matern_cellwise', 'file': 'prior-moments.npz', 'shape': [nx, nz],
                      'spacing': [dx, dz], 'lengths': problem['ranges'].tolist(),
                      'std_floors': list(STD_FLOORS), 'boundary': 'robin'}
        covariance_note = ('Matern nu=1 prior with the carried chain ensemble\'s cellwise mean and '
                           f'std (floors {STD_FLOORS}); chain ranges; Robin boundary')
    n_starts, start_seed = getattr(args, 'starts', 0), getattr(args, 'start_seed', 0)
    start_file = getattr(args, 'start_models', None)
    if n_starts and start_file is not None:
        raise ValueError('Use either --starts or --start-models')
    count = n_starts or 1
    starts, labels = {}, None
    start_note = 'start at the prior mean'
    if start_file is not None:
        with np.load(start_file, allow_pickle=False) as saved:
            models, labels = saved['models'].astype(float), [str(label) for label in saved['labels']]
        if models.ndim != 2 or models.shape[1] != center.size or len(labels) != len(models):
            raise ValueError('Start models must be k x n with one label each')
        count = len(models)
        starts = {'start_models': models}
        start_note = f'{count} starts ({"; ".join(labels)}); prior and data shared'
    elif n_starts:
        ensemble = problem['prior_ensemble']
        if not 0 < n_starts <= ensemble.shape[1]:
            raise ValueError('--starts must lie between 1 and the carried ensemble size')
        members = np.sort(np.random.default_rng(start_seed).choice(ensemble.shape[1], n_starts, replace=False))
        starts = {'start_models': ensemble[:, members].T, 'start_members': members}
        labels = [f'member {member}' for member in members]
        if getattr(args, 'include_mean_start', False):
            # Member 0 keeps the single-start behaviour (the prior mean); the others start from members.
            starts['start_models'] = np.vstack([center, starts['start_models']])
            labels = ['prior mean'] + labels
            count = n_starts + 1
        start_note = (f'{n_starts} starts from carried-ensemble members {members.tolist()} '
                      f'(seed {start_seed}); prior and data shared')
    atomic_npz(output / 'inputs.npz', prior_centers=np.tile(center, (count, 1)),
               measurement_noise=np.zeros((count, len(observations))),
               targets=np.tile(observations, (count, 1)),
               observations=observations, variance=variance / weight, mask=problem['mask'], **starts,
               **{key: value for key, value in problem.items() if key.startswith('well_')})
    study = {**source, 'method': 'chain_laplace_map', 'index': problem['index'], 'count': count,
             'dims': [nx, ny, nz], 'spacing_ft': [dx, dy, dz], 'shift_ft': problem['shift_ft'],
             'tools': [list(tool) for tool in problem['tools']],
             'prior': prior_spec, 'likelihood_weight': weight,
             'anisotropy': problem['anisotropy'],
             'anisotropy_state_moments': problem['anisotropy_moments'].tolist(),
             'ratio_min': 1., 'rh_limits': [-1., float(np.log(40.))], 'log_rv_cap': float(np.log(40.)),
             'parameterization': 'raw Gaussian state; Rh clipped only in physical forward map; '
                                 f"anisotropy {problem['anisotropy']}",
             'covariance_note': covariance_note,
             'randomization_note': (f'MAP: unperturbed observations, Cd / {weight:g} (2D share of the data '
                                    f'energy), {start_note}'),
             'save_jacobian': True,
             'chain': {'run': str(problem_path.parent), 'problem': str(problem_path),
                       'problem_sha256': digest(problem_path), 'probe': problem['probe'],
                       'window_movement': [int(m) for m in problem['movement']]},
             'source_study': str(args.source.resolve()),
             'source_sha256': {name: digest(ROOT / name) for name in MODULES}}
    prior_scale = getattr(args, 'prior_scale', None)
    if prior_scale not in (None, 1, 1.):
        if not prior_scale > 0:
            raise ValueError('The prior scale must be positive')
        study['prior']['scale'] = float(prior_scale)
        study['covariance_note'] += f'; every prior std multiplied by {prior_scale:g}'
    if 'graded_dz' in problem:
        study['graded_dz'] = [int(n) for n in problem['graded_dz']]
    if n_starts:
        study['starts'] = {'count': count, 'seed': start_seed, 'labels': labels,
                           'members': starts['start_members'].tolist()}
    elif start_file is not None:
        study['starts'] = {'count': count, 'labels': labels, 'file': str(Path(start_file).resolve()),
                           'file_sha256': digest(start_file)}
    study.pop('reused_inputs_sha256', None)
    study.pop('cascade', None)
    study['inputs_sha256'] = digest(output / 'inputs.npz')
    atomic_json(output / 'study.json', study)
    write_array(output, study)
    print(f"Prepared the Laplace MAP problem for chain index {problem['index']} ({start_note}): {output}")


def write_array(output, study):
    """One array element per member; its allocation hosts all candidate solver steps.

    Each member holds candidates x ranks Slurm tasks of `cpus` physical cores;
    every candidate evaluation launches one `srun --exact` step of `ranks`
    tasks inside it. The Slurm queue paces how many members run at once.
    """
    simulator = study['simulator']
    resources = study.get('resources', {'memory_gib': 8, 'walltime': '04:00:00'})
    tasks = simulator.get('concurrency', 1) * simulator.get('ranks', 1)
    # Slurm counts hyperthreads as CPUs here (2 per core). Physical-core steps
    # (`srun --hint=nomultithread --cpus-per-task=C`) each consume 2C CPUs, so the
    # allocation must hold 2C per task or concurrent candidate steps queue.
    cpus = simulator['cpus'] * (2 if simulator.get('physical_cores') else 1)
    script = f'''#!/bin/bash
#SBATCH --job-name=rml-index{study['index']}
#SBATCH --partition=comp
#SBATCH --account=energytech
#SBATCH --qos=normal
#SBATCH --array=0-{study['count'] - 1}
#SBATCH --nodes=1
#SBATCH --ntasks={tasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={resources['memory_gib']}G
#SBATCH --time={resources['walltime']}
#SBATCH --output=member-%A_%a.log
set -euo pipefail
: "${{RML_PROJECT:?Set RML_PROJECT to the deployed code directory}}"
: "${{RML_RUN:?Set RML_RUN to the prepared run directory}}"
: "${{RML_PYTHON:?Set RML_PYTHON to the project-local HPC Python}}"
export PYTHONPATH="$RML_PROJECT:$RML_PROJECT/pxwf-src${{PYTHONPATH:+:$PYTHONPATH}}"
export PXWF_SRC="$RML_PROJECT/pxwf-src"
# Solver threads are set per srun step by the worker; keep Python single-threaded.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
ulimit -s unlimited 2>/dev/null || true
cd "$RML_PROJECT"
exec "$RML_PYTHON" -B -u pilot_rml_index45.py "${{RML_ACTION:-member}}" --run-dir "$RML_RUN" --member "$SLURM_ARRAY_TASK_ID"
'''
    (output / 'rml-array.sbatch').write_text(script)


def read_study(directory):
    study = json.loads((directory / 'study.json').read_text())
    if digest(directory / 'inputs.npz') != study['inputs_sha256']:
        raise ValueError('Prepared inputs changed')
    if any(digest(ROOT / name) != expected for name, expected in study['source_sha256'].items()):
        raise ValueError('Implementation changed since preparation; use the matching code bundle')
    with np.load(directory / 'inputs.npz', allow_pickle=False) as saved:
        inputs = {key: saved[key].copy() for key in saved.files}
    return study, inputs


def evaluation_setup(study, inputs, output):
    """Simulator adapter, well and raw-to-physical map for a prepared study, inside an allocation."""
    from esmda_remote2d import SimBackendConfig, make_remote2d
    import run_DA_sequential as sequential

    if study.get('execution') == 'node':
        # A plain compute node (no Slurm): single-rank standalone solves, launched by node-members.sh.
        if not os.environ.get('RML_NODE'):
            raise RuntimeError('Node studies run under the node launcher (RML_NODE=1)')
    else:
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run solver evaluations within a Slurm allocation')
        if int(os.environ.get('SLURM_CPUS_PER_TASK', '1')) < study['simulator']['cpus']:
            raise RuntimeError('Slurm task has fewer CPUs than the prepared solver configuration')

    expand, contract, solver_dims, solver_spacing = grid_maps(study)
    sequential.dims = tuple(solver_dims)
    sequential.dX, sequential.dY, sequential.dZ = solver_spacing
    sequential.min_ratio, sequential.max_ratio = study['ratio_min'], study['ratio_max']
    sequential.log_rv_cap = study['log_rv_cap']
    sequential.anisotropy = study.get('anisotropy', 'probit')
    config = SimBackendConfig(**study['simulator'])
    if config.backend != 'local':
        raise ValueError('Array members evaluate inside their own allocation (local backend)')
    adapter = make_remote2d(config, sequential.dims, *solver_spacing, study['tools'])
    adapter.jobs_dir = str(output / 'jobs')
    if study.get('clean_jobs'):
        # Solver scratch (GBs per evaluation) is dropped as soon as the results are in memory.
        import shutil

        def cleaned(method):
            def call(*arguments, **keywords):
                try:
                    return method(*arguments, **keywords)
                finally:
                    shutil.rmtree(adapter.jobs_dir, ignore_errors=True)
            return call

        adapter.run_pred_multi = cleaned(adapter.run_pred_multi)
        adapter.run_grad_multi = cleaned(adapter.run_grad_multi)
    well = {key[5:]: value for key, value in inputs.items() if key.startswith('well_')}
    cells = solver_dims[0] * solver_dims[2]
    lower, upper = study['rh_limits']

    def physical_parameters(raw):
        """Physical fields of a raw model state, on the solver grid (model cells expanded)."""
        physical = expand(np.asarray(raw, dtype=float))
        physical[:cells] = np.clip(physical[:cells], lower, upper)
        return sequential._build_param_dict(physical)

    physical_parameters.expand, physical_parameters.contract = expand, contract
    return adapter, well, physical_parameters


def grid_maps(study):
    """Model-to-solver grid maps: identity for uniform cells, or a graded vertical grid.

    ``study['graded_dz']`` lists, per model row, how many 1-ft solver rows it spans
    (top to bottom, the same for every column). Models are expanded row-wise onto
    the uniform 1-ft solver grid and Jacobians contracted by summing those rows,
    which is exact for the piecewise-constant expansion. Returns (expand,
    contract, solver dims, solver spacing).
    """
    nx, ny, nz = study['dims']
    dx, dy, dz = study['spacing_ft']
    counts = study.get('graded_dz')
    if counts is None:
        return (lambda raw: np.array(raw, dtype=float, copy=True)), (lambda jacobian: jacobian), \
            (nx, ny, nz), (dx, dy, dz)
    counts = np.asarray(counts, dtype=int)
    if counts.shape != (nz,) or np.any(counts < 1):
        raise ValueError('graded_dz needs one positive row count per model row')
    rows = int(counts.sum())
    starts = np.r_[0, np.cumsum(counts)[:-1]]

    def expand(raw):
        return np.repeat(np.asarray(raw, dtype=float).reshape(2, nx, nz), counts, axis=2).ravel()

    def contract(jacobian):
        jacobian = np.asarray(jacobian, dtype=float)
        return np.add.reduceat(jacobian.reshape(jacobian.shape[0], 2, nx, rows), starts, axis=3).reshape(
            jacobian.shape[0], 2 * nx * nz)

    # The worker requires dY == dX; 1-ft solver rows.
    return expand, contract, (nx, ny, rows), (dx, dy, 1.)


def phase_flags(inputs):
    from esmda_core import phase_channels
    from esmda_remote2d import UDAR_COMPONENTS

    return phase_channels(inputs['mask'], UDAR_COMPONENTS)


def forward_evaluator(study, inputs, output):
    """Forward-only 2D predictions (retained channels) for raw states; NaN rows mark failed solves.

    Predicted phases are moved to the branch nearest the observed ones, so
    residuals never jump by 360 degrees.
    """
    from esmda_core import align_phases

    adapter, well, physical_parameters = evaluation_setup(study, inputs, output)
    phase = phase_flags(inputs)

    def predict_batch(raws):
        parameters = [physical_parameters(np.asarray(raw, dtype=float))[0] for raw in raws]
        predictions = np.asarray(adapter.run_pred_multi(parameters, study['index'], study['shift_ft'], well),
                                 dtype=float)
        return align_phases(predictions[:, inputs['mask']], inputs['observations'], phase, axis=1)

    return predict_batch


def model_evaluator(study, inputs, output):
    from esmda_core import align_phases
    import run_DA_sequential as sequential

    adapter, well, physical_parameters = evaluation_setup(study, inputs, output)
    if not adapter.backend().trusts_ad('2D'):
        raise RuntimeError('A trusted exact-adjoint build is required')
    expand, contract = physical_parameters.expand, physical_parameters.contract
    cells = sequential.dims[0] * sequential.dims[2]     # solver-grid cells
    lower, upper = study['rh_limits']
    phase = phase_flags(inputs)

    def compose(raw, rh, latent, prediction, j_rh, j_rv):
        solver_raw = expand(raw)
        jacobian = sequential.optimizer_jacobian(
            j_rh, j_rv, latent, sequential.min_ratio, sequential.max_ratio,
            log_rv_cap=sequential.log_rv_cap, log_rh=rh, anisotropy=sequential.anisotropy)
        jacobian[:, :cells] *= ((solver_raw[:cells] > lower) & (solver_raw[:cells] < upper))[None, :]
        # Phases on the observed branch: residuals stay in [-180, 180) degrees.
        return (align_phases(prediction[inputs['mask']], inputs['observations'], phase),
                contract(jacobian[inputs['mask']]))

    def evaluate_batch(raws):
        """Forward + Jacobian for several raw states at once; None marks a failed solve."""
        converted = [physical_parameters(raw) for raw in raws]
        results = adapter.run_grad_multi([parameters for parameters, _, _ in converted],
                                         study['index'], study['shift_ft'], well)
        return [None if j_rh is None else compose(raw, rh, latent, prediction, j_rh, j_rv)
                for raw, (_, rh, latent), (prediction, j_rh, j_rv) in zip(raws, converted, results)]

    def evaluate(raw):
        result = evaluate_batch([raw])[0]
        if result is None:
            raise RuntimeError('Solver evaluation failed')
        return result

    evaluate.batch = evaluate_batch
    return evaluate


def member(args):
    directory = args.run_dir.resolve()
    study, inputs = read_study(directory)
    if not 0 <= args.member < study['count']:
        raise ValueError('Member ID outside prepared range')
    output = directory / f'member-{args.member:03d}'
    output.mkdir(exist_ok=True)
    with (output / 'member.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = output / 'result.json'
        identity = digest(directory / 'study.json')
        if report.exists() and json.loads(report.read_text()).get('reason') in TERMINAL_REASONS:
            if json.loads(report.read_text())['identity'] != identity or not (output / 'fit.npz').exists():
                raise ValueError('Finished member has inconsistent metadata or missing fit')
            print(f'Member {args.member} already finished')
            return
        prior = load_prior(study, directory)
        try:
            evaluate = model_evaluator(study, inputs, output)
            if study['optimizer_method'] == LM_OPTIMIZER:
                fit = optimize_member_lm(
                    prior, inputs['prior_centers'][args.member], inputs['targets'][args.member],
                    inputs['variance'], evaluate.batch, checkpoint=output / 'checkpoint.npz',
                    identity=identity,
                    start=inputs['start_models'][args.member] if 'start_models' in inputs else None,
                    **study['optimizer'])
            else:
                fit = optimize_member(
                    prior, inputs['prior_centers'][args.member], inputs['targets'][args.member],
                    inputs['variance'], evaluate, checkpoint=output / 'checkpoint.npz',
                    identity=identity, **study['optimizer'])
            if study.get('save_jacobian'):
                # Jacobian at the solution for the Laplace approximation of the posterior.
                prediction, jacobian = evaluate(fit['model'])
                atomic_npz(output / 'laplace.npz', model=fit['model'], prediction=prediction,
                           jacobian=jacobian)
            atomic_npz(output / 'fit.npz', model=fit.pop('model'), z=fit.pop('z'),
                       prediction=fit.pop('prediction'))
            fit.update(member=args.member, identity=identity)
            atomic_json(report, fit)
        except Exception as error:
            atomic_json(report, {'member': args.member, 'converged': False,
                                 'reason': 'execution_failure', 'error': str(error),
                                 'identity': identity})
            raise
        print(json.dumps({key: fit[key] for key in ('reason', 'converged', 'accepted_iterations',
                                                    'evaluations_this_attempt', 'seconds_this_attempt')}))
        if fit['reason'] not in TERMINAL_REASONS:
            # Resumable (walltime budget): rerun the same array element to continue.
            raise SystemExit(3)


def forward_start(args):
    """Forward-only 2D prediction (phases aligned to the observations) of one member's start model."""
    directory = args.run_dir.resolve()
    study, inputs = read_study(directory)
    if not 0 <= args.member < study['count'] or 'start_models' not in inputs:
        raise ValueError('Member ID outside prepared range, or no start models')
    output = directory / f'member-{args.member:03d}'
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    adapter, well, physical_parameters = evaluation_setup(study, inputs, output)
    full = np.asarray(adapter.run_pred_multi([physical_parameters(inputs['start_models'][args.member])[0]],
                                             study['index'], study['shift_ft'], well), dtype=float)[0]
    from esmda_core import align_phases
    prediction = align_phases(full[inputs['mask']], inputs['observations'], phase_flags(inputs))
    atomic_npz(output / 'forward.npz', prediction=prediction, prediction_full=full,
               model=inputs['start_models'][args.member])
    atomic_json(output / 'forward.json', {'member': args.member, 'identity': digest(directory / 'study.json'),
                                          'seconds': time.monotonic() - started,
                                          'finite': bool(np.isfinite(prediction).all())})
    print(f'Member {args.member}: forward in {time.monotonic() - started:.1f} s')


def gradient_check(args):
    directory = args.run_dir.resolve()
    study, inputs = read_study(directory)
    if not 0 <= args.member < study['count']:
        raise ValueError('Member ID outside prepared range')
    output = directory / f'gradient-check-{args.member:03d}'
    output.mkdir(exist_ok=True)
    prior = load_prior(study, directory)
    evaluate = model_evaluator(study, inputs, output)
    rng = np.random.default_rng(10000 + args.member)
    z = rng.normal(0., .01, len(prior.mean))
    direction = rng.normal(size=len(z))
    direction /= np.linalg.norm(direction)
    def objective(point):
        return objective_gradient(point, inputs['prior_centers'][args.member],
                                  inputs['targets'][args.member], inputs['variance'],
                                  prior, evaluate)
    baseline, gradient, _, _, _ = objective(z)
    repeated = objective(z)[0]
    analytic = float(gradient @ direction)
    rows = []
    for step in (1e-1, 3e-2, 1e-2, 3e-3):
        plus, minus = objective(z + step * direction)[0], objective(z - step * direction)[0]
        finite = float((plus - minus) / (2 * step))
        rows.append({'step': step, 'finite_difference': finite, 'analytic': analytic,
                     'objective_plus': plus, 'objective_minus': minus,
                     'relative_error': abs(finite - analytic) / max(abs(finite), abs(analytic), 1e-8)})
    passed = sum(row['relative_error'] < .01 for row in rows) >= 2
    atomic_json(output / 'result.json', {'passed': passed, 'checks': rows,
                                        'baseline_objective': baseline,
                                        'repeat_objective_difference': repeated - baseline,
                                        'solver_tolerance': study['simulator']['solver_tolerance']})
    print(json.dumps(rows, indent=2))
    if not passed:
        raise SystemExit(2)


def collect(args):
    directory = args.run_dir.resolve()
    study, inputs = read_study(directory)
    models, predictions, summaries = [], [], []
    complete = True
    for member_id in range(study['count']):
        output = directory / f'member-{member_id:03d}'
        report = output / 'result.json'
        summary = json.loads(report.read_text()) if report.exists() else {
            'member': member_id, 'converged': False, 'reason': 'missing'}
        if report.exists() and summary['identity'] != digest(directory / 'study.json'):
            raise ValueError('Member result belongs to different study')
        complete &= summary.get('reason') in TERMINAL_REASONS
        if (output / 'fit.npz').exists() and summary['reason'] != 'execution_failure':
            with np.load(output / 'fit.npz', allow_pickle=False) as saved:
                model, prediction = saved['model'].copy(), saved['prediction'].copy()
            # Original data against the original (uninflated) noise; randomized
            # targets against the variance they were drawn with.
            summary['original_data_misfit'] = float(np.mean(
                (prediction - inputs['observations']) ** 2
                / inputs.get('original_variance', inputs['variance'])))
            summary['randomized_data_misfit'] = float(np.mean(
                (prediction - inputs['targets'][member_id]) ** 2 / inputs['variance']))
            cells = len(model) // 2
            summary['rh_cap_fraction'] = float(np.mean(
                (model[:cells] <= study['rh_limits'][0]) | (model[:cells] >= study['rh_limits'][1])))
            models.append(model)
            predictions.append(prediction)
        else:
            models.append(np.full(len(inputs['prior_centers'][0]), np.nan))
            predictions.append(np.full_like(inputs['observations'], np.nan))
        summaries.append(summary)
    models, predictions = np.column_stack(models), np.column_stack(predictions)
    atomic_npz(directory / 'rml-ensemble.npz', raw_models=models, predictions=predictions,
               member_ids=np.arange(study['count']),
               converged=np.array([row['converged'] for row in summaries]))
    reasons = {}
    for summary in summaries:
        reasons[summary.get('reason', 'missing')] = reasons.get(summary.get('reason', 'missing'), 0) + 1
    report = {'complete': bool(complete), 'reasons': reasons, 'members': summaries}
    if complete:
        report['average_member_misfit'] = float(np.mean([s['original_data_misfit'] for s in summaries]))
        report['mean_prediction_misfit'] = float(np.mean(
            (predictions.mean(axis=1) - inputs['observations']) ** 2 / inputs['variance']))
        physical = models.copy()
        cells = models.shape[0] // 2
        physical[:cells] = np.clip(physical[:cells], *study['rh_limits'])
        prior_physical = inputs['prior_centers'].T.copy()
        prior_physical[:cells] = np.clip(prior_physical[:cells], *study['rh_limits'])
        atomic_npz(directory / 'moments.npz', raw_mean=models.mean(axis=1),
                   raw_std=models.std(axis=1, ddof=1), physical_mean=physical.mean(axis=1),
                   physical_std=physical.std(axis=1, ddof=1),
                   prior_physical_mean=prior_physical.mean(axis=1),
                   prior_physical_std=prior_physical.std(axis=1, ddof=1))
        if args.verify_mean:
            output = directory / 'mean-model-check'
            output.mkdir(exist_ok=True)
            prediction, _ = model_evaluator(study, inputs, output)(models.mean(axis=1))
            atomic_npz(output / 'prediction.npz', prediction=prediction)
            report['mean_model_misfit'] = float(np.mean(
                (prediction - inputs['observations']) ** 2 / inputs['variance']))
        else:
            report['mean_model_misfit'] = None
        plot_ensemble(directory, study, physical, prior_physical)
    atomic_json(directory / 'summary.json', report)
    print(json.dumps({'complete': bool(complete), 'converged': sum(s['converged'] for s in summaries),
                      'reasons': reasons, 'total': study['count']}))


def plot_ensemble(directory, study, physical, prior_physical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nx, _, nz = study['dims']
    cells = nx * nz
    extent = [study['shift_ft']['x'], study['shift_ft']['x'] + nx * study['spacing_ft'][0],
              study['shift_ft']['z'] + nz * study['spacing_ft'][2], study['shift_ft']['z']]
    side = int(np.ceil(np.sqrt(min(physical.shape[1], 25))))
    figure, axes = plt.subplots(side, side, figsize=(16, 16), constrained_layout=True, squeeze=False)
    for axis in axes.flat[physical.shape[1]:]:
        axis.set_visible(False)
    for member_id, axis in zip(range(physical.shape[1]), axes.flat):
        image = axis.imshow((physical[:cells, member_id] / np.log(10.)).reshape(nx, nz).T,
                            extent=extent, aspect='auto', vmin=-1 / np.log(10.),
                            vmax=np.log10(40.), cmap='viridis')
        axis.set_title(f'RML {member_id}')
    figure.colorbar(image, ax=axes.ravel().tolist(), label='log10 Rh [ohm m]')
    figure.savefig(directory / 'rml-realizations.png', dpi=120)
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for row, (label, fields) in enumerate((('Prior', prior_physical), ('RML', physical))):
        rh = fields[:cells] / np.log(10.)
        for column, (name, values) in enumerate((('mean', rh.mean(axis=1)),
                                                ('std', rh.std(axis=1, ddof=1)))):
            image = axes[row, column].imshow(values.reshape(nx, nz).T, extent=extent,
                                             aspect='auto', cmap='viridis',
                                             vmin=0 if name == 'std' else -1 / np.log(10.),
                                             vmax=None if name == 'std' else np.log10(40.))
            axes[row, column].set_title(f'{label} log10 Rh {name}')
            figure.colorbar(image, ax=axes[row, column])
    figure.savefig(directory / 'prior-rml-moments.png', dpi=150)
    plt.close(figure)


def add_code(archive, modules=MODULES):
    """Project modules, HPC requirements and the pxwf source tree under ``code/``."""
    source = ROOT.parent / 'Dashboard/src/pxwf'
    if not source.is_dir():
        raise FileNotFoundError(source)
    for name in modules:
        archive.add(ROOT / name, arcname=f'code/{name}')
    archive.add(ROOT / 'rml-hpc-requirements.txt', arcname='code/rml-hpc-requirements.txt')
    for path in sorted(source.rglob('*.py')):
        archive.add(path, arcname=str(Path('code/pxwf-src/pxwf') / path.relative_to(source)))


def bundle(args):
    directory = args.run_dir.resolve()
    read_study(directory)
    destination = directory / 'deployment.tar.gz'
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(destination, 'w:gz') as archive:
        add_code(archive)
        for name in ('study.json', 'inputs.npz', 'rml-array.sbatch', 'prior-ensemble.npz',
                     'prior-moments.npz', 'prior-precision.npz'):
            if not (directory / name).exists():
                continue
            archive.add(directory / name, arcname=f'run/{name}')
    atomic_json(directory / 'deployment.json', {'archive': destination.name,
                                              'sha256': digest(destination),
                                              'source_sha256': {name: digest(ROOT / name) for name in MODULES}})
    print(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    prepare_parser.add_argument('--index', type=int, default=45)
    prepare_parser.add_argument('--count', type=int, default=25)
    prepare_parser.add_argument('--seed', type=int, default=2045)
    prepare_parser.add_argument('--cpus', type=int, default=8,
                                help='Physical cores per MPI rank (OpenMP threads)')
    prepare_parser.add_argument('--ranks', type=int, default=2,
                                help='MPI ranks per solve (fwd2D splits the two receiver groups)')
    prepare_parser.add_argument('--mkl-threads', type=int, default=8)
    prepare_parser.add_argument('--mkl-cbwr', default='AVX2,STRICT',
                                help='Pinned MKL code path (part of the forward map identity)')
    prepare_parser.add_argument('--binary-dir', default='/cluster/home/krfo/adjoint-2d-v1')
    prepare_parser.add_argument('--oneapi', default='')
    prepare_parser.add_argument('--hdfql', default='')
    prepare_parser.add_argument('--solver-tolerance', type=float, default=1e-6)
    prepare_parser.add_argument('--reuse-inputs', type=Path,
                                help='Reuse byte-identical randomizations from an earlier run')
    prepare_parser.add_argument('--candidates', type=int, default=4,
                                help='Damping candidates evaluated in parallel per LM round')
    prepare_parser.add_argument('--initial-trust', type=float, default=.3,
                                help='Initial limit on the largest raw-state change per step')
    prepare_parser.add_argument('--max-trust', type=float, default=2.)
    prepare_parser.add_argument('--min-trust', type=float, default=1e-3)
    prepare_parser.add_argument('--max-iterations', type=int, default=50)
    prepare_parser.add_argument('--max-rounds', type=int, default=4,
                                help='Consecutive rejected candidate rounds before nonlinearity_limited')
    prepare_parser.add_argument('--max-seconds', type=float, default=3300.)
    prepare_parser.add_argument('--stagnation-tolerance', type=float, default=1e-3)
    prepare_parser.add_argument('--stagnation-iterations', type=int, default=2)
    prepare_parser.add_argument('--relative-gradient-tolerance', type=float, default=1e-3)
    prepare_parser.add_argument('--memory-gib', type=int, default=48)
    prepare_parser.add_argument('--walltime', default='01:00:00')
    cascade_parser = sub.add_parser('prepare-cascade',
                                    help='RML for the remaining likelihood after a tempered cascade')
    cascade_parser.add_argument('--source', type=Path, default=ROOT / 'runs/rml-index45-lm-20261019b',
                                help='Prepared RML study supplying the problem, optimizer and solver settings')
    cascade_parser.add_argument('--selection', type=Path, required=True,
                                help='cascade_handoff.py selection.json')
    cascade_parser.add_argument('--noise-seed', type=int, default=4511)
    cascade_parser.add_argument('--prior', choices=['cascade', 'original', 'cascade-prior'],
                                default='cascade',
                                help='cascade: SpatialPrior of the cascade ensemble, tempered data; '
                                     'original: original prior and untempered data, cascade members '
                                     'used only as starting models; cascade-prior: as original, but '
                                     'with the Matern prior the cascade drew from and fresh centres')
    laplace_parser = sub.add_parser('prepare-laplace',
                                    help='MAP problem for a sequential-chain gate point (Laplace stage)')
    laplace_parser.add_argument('--problem', type=Path, required=True,
                                help='laplace-problem-<index>.npz exported by sequential_matern_1d.py')
    laplace_parser.add_argument('--source', type=Path, default=ROOT / 'runs/rml-index45-lm-20261019b',
                                help='Prepared RML study supplying the optimizer and solver settings')
    laplace_parser.add_argument('--binary-dir', default=None, help='Override the source study\'s fwd2D build')
    laplace_parser.add_argument('--orientation', choices=['gradient', 'bedding'], default=None,
                                help='Override the source study\'s material orientation')
    laplace_parser.add_argument('--starts', type=int, default=0,
                                help='Multi-start test: this many members, each starting from a distinct '
                                     'carried-ensemble member (default 0: one member from the prior mean)')
    laplace_parser.add_argument('--start-seed', type=int, default=0, help='Selects the starting members')
    laplace_parser.add_argument('--optimizer-setting', action='append', default=None, metavar='KEY=VALUE',
                                help='Override one optimizer setting of the source study (repeatable)')
    laplace_parser.add_argument('--node-solver', type=int, nargs=2, default=None, metavar=('THREADS', 'CONCURRENCY'),
                                help='Prepare for a plain compute node: single-rank solves with THREADS threads, '
                                     'CONCURRENCY candidate solves at once, scratch removed per evaluation')
    laplace_parser.add_argument('--hdfql', default=None, help='Directory with libHDFql.so on the execution host')
    laplace_parser.add_argument('--include-mean-start', action='store_true',
                                help='With --starts: member 0 starts from the prior mean, the others from members')
    laplace_parser.add_argument('--start-models', type=Path, default=None,
                                help='Multi-start test from explicit models (npz: models k x n, labels)')
    laplace_parser.add_argument('--likelihood-weight', type=float, default=None,
                                help="Replace the problem's likelihood weight (Cd / weight)")
    laplace_parser.add_argument('--prior-scale', type=float, default=None,
                                help='Multiply every prior standard deviation by this factor (>1 weakens the prior)')
    laplace_parser.add_argument('--prior', choices=['carried', 'matern'], default='carried',
                                help="carried: the exported problem's prior; matern: the chain's stationary "
                                     'Matern prior')
    commands = [('prepare', prepare_parser), ('prepare-cascade', cascade_parser),
                ('prepare-laplace', laplace_parser),
                ('member', sub.add_parser('member')),
                ('forward-start', sub.add_parser('forward-start')),
                ('gradient-check', sub.add_parser('gradient-check')),
                ('collect', sub.add_parser('collect')), ('bundle', sub.add_parser('bundle'))]
    for name, command in commands:
        command.add_argument('--run-dir', type=Path, required=True)
        if name in ('member', 'gradient-check', 'forward-start'):
            command.add_argument('--member', type=int, required=True)
        if name == 'collect':
            command.add_argument('--verify-mean', action='store_true',
                                 help='Evaluate the raw mean model on an allocated compute node')
    args = parser.parse_args(argv)
    if args.command == 'prepare' and (
            args.count != 25 or min(args.cpus, args.ranks, args.mkl_threads, args.candidates,
                                    args.max_iterations, args.max_rounds, args.stagnation_iterations,
                                    args.memory_gib) < 1
            or not 0 < args.min_trust <= args.initial_trust <= args.max_trust
            or min(args.max_seconds, args.solver_tolerance, args.stagnation_tolerance,
                   args.relative_gradient_tolerance) <= 0
            or not np.isfinite(args.max_seconds + args.max_trust + args.solver_tolerance)):
        parser.error('Require 25 members and positive finite resource/optimizer settings')
    {'prepare': prepare, 'prepare-cascade': prepare_cascade, 'prepare-laplace': prepare_laplace,
     'member': member, 'forward-start': forward_start,
     'gradient-check': gradient_check, 'collect': collect, 'bundle': bundle}[args.command](args)


if __name__ == '__main__':
    main()
