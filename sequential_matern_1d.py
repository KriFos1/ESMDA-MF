"""Sequential 1D ES-MDA with the Matern prior: each point's posterior is the next point's prior.

Every logging point spends the full likelihood (or the ``--likelihood-weight``
fraction of it) in ``--mda-steps`` ES-MDA steps with the 1D forward only: no 0D stage, no adaptive localization,
no misfit gate and no 2D stage. The ensemble starts as Matern (nu = 1) draws at
the first index. When the local window moves, kept cells carry the assimilated
ensemble and the entered cells are redrawn from the Matern GMRF conditionally
on the kept cells (``MaternPrior.condition_entered``). Checkpoints follow the
sequential ``inversion_results_assim_<index>.pkl`` layout, so ``plot_curtain``
and the benchmark RMSE scoring read them unchanged.

python -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
  --output-dir runs/gated-enif-t25-bedding --stop 281 --ensemble-size 250 --seed 2031 \
  --mda-steps 10 --tempering linear-inflation --anisotropy log_clipped \
  --anisotropy-mean 0.5 --anisotropy-std 0.5 --memory-ft 50 --gate-threshold 25 \
  --on-gate stop --analysis enif --two-d-prior ensemble_precision \
  --directional-sigma-floor 0.5 0.25


"""

import argparse
import json
import os
import re
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from chain_laplace import export_problem
from esmda_core import (LocalWindow, align_phases, clamp_log_rh, log_rv_from_latent, noise_floor_mask,
                        observation_vectors,
                        phase_channels)
from esmda_remote2d import UDAR_COMPONENTS
from pilot_2d_only_index31 import RH_CORRELATION_X_FT, RH_CORRELATION_Z_FT, cap_rh
from pilot_cascade_laplace_index34 import (cascade_inflation, ensemble_diagnostics,
                                           matern_reference, member_misfit_summary)
from enif_mda import EnIFProbe
from increment_prior import ensemble_student_weights, increment_precision
from sparse_precision import EnsemblePrecisionPrior, conditional_regression
import run_DA_sequential as sequential
from transition_mda_study import DEFAULT_REFERENCE, reference_truth, rh_rmse, update

ROOT = Path(__file__).resolve().parent
RH_CAP_OHMM = 40.
RV_CAP_OHMM = 40.
PROPAGATION_RADIUS = 2      # cells: neighbourhood of the ridge precision used by --lateral-update ridge


def entered_cells(nx, nz, movement):
    """Boolean (nx, nz) mask of the cells that enter the window for a (dx, dz) cell move."""
    entered = np.zeros((nx, nz), dtype=bool)
    move_x, move_z = (int(step) for step in movement)
    if move_x > 0:
        entered[max(nx - move_x, 0):] = True
    elif move_x < 0:
        entered[:min(-move_x, nx)] = True
    if move_z > 0:
        entered[:, max(nz - move_z, 0):] = True
    elif move_z < 0:
        entered[:, :min(-move_z, nz)] = True
    return entered


def reindex_window(state, nx, nz, movement):
    """Shift the window state by whole cells; cells without a source keep the edge value."""
    members = state.shape[1]
    x = np.clip(np.arange(nx) + int(movement[0]), 0, nx - 1)
    z = np.clip(np.arange(nz) + int(movement[1]), 0, nz - 1)
    fields = state.reshape(2, nx, nz, members)[:, x][:, :, z]
    return fields.reshape(2 * nx * nz, members).copy()


def relax_to_prior(prior, state, rho, rng, rho_mean=None):
    """Discrete Ornstein-Uhlenbeck step toward the Matern prior.

    x <- mu + rho (x - mu) + sqrt(1 - rho^2) L eps keeps N(mu, L L^T) invariant, so a
    carried ensemble forgets assimilated information geometrically (memory
    length -dMD / ln rho) and can neither collapse below the data-versus-forgetting
    equilibrium nor drift out of the prior support; the injected perturbations
    carry the full Matern structure.

    ``rho`` may be a scalar, one factor per cell (shared by the ln Rh and latent blocks) or one per state
    entry. With ``rho_mean`` (same shapes) the ensemble mean relaxes to the prior mean at its own rate while
    the anomalies relax with ``rho``: x <- mu + rho_mean (m - mu) + rho (x - m) + sqrt(1 - rho^2) L eps,
    m the ensemble mean, so the carried local mean (e.g. a resolved transition) outlives the spread.
    """
    def per_state(value):
        value = np.asarray(value, dtype=float)
        if not np.all((0. <= value) & (value <= 1.)):
            raise ValueError('Require 0 <= rho <= 1')
        if value.ndim and len(value) == len(prior.mean) // 2:
            value = np.tile(value, 2)
        return value[:, None] if value.ndim else value

    rho = per_state(rho)
    state = np.asarray(state, dtype=float)
    if rho_mean is None:
        if np.all(rho == 1.):
            return np.array(state, copy=True)
        offset = state - prior.mean[:, None]
        noise = prior.color(rng.normal(size=offset.shape))
        return prior.mean[:, None] + rho * offset + np.sqrt(1. - rho ** 2) * noise
    rho_mean = per_state(rho_mean)
    if np.all(rho == 1.) and np.all(rho_mean == 1.):
        return np.array(state, copy=True)
    mean = state.mean(axis=1, keepdims=True)
    anomaly = state - mean
    noise = prior.color(rng.normal(size=state.shape))
    return prior.mean[:, None] + rho_mean * (mean - prior.mean[:, None]) + rho * anomaly \
        + np.sqrt(1. - rho ** 2) * noise


class TwinTruth:
    """Synthetic truth of a twin experiment: one draw of the Matern prior on a grid that covers every
    window of the run (same lattice as the moving window, plus a margin), observed through the 1D operator.

    The data are then consistent with the prior, the forward model and the noise model by construction,
    so any loss of calibration is produced by the update scheme (or the carry), not by model error.
    """
    MARGIN = 20                 # cells around the union of the windows (keeps the grid boundary out of them)

    def __init__(self, origin, spacing, fields, seed):
        self.origin, self.spacing, self.fields, self.seed = tuple(origin), tuple(spacing), fields, int(seed)

    @classmethod
    def draw(cls, manifest, shifts, anisotropy, moments, seed):
        nx, ny, nz = manifest['dims']
        dx, _, dz = manifest['spacing_ft']
        x0 = min(shift['x'] for shift in shifts) - cls.MARGIN * dx
        z0 = min(shift['z'] for shift in shifts) - cls.MARGIN * dz
        size_x = int(round((max(shift['x'] for shift in shifts) - x0) / dx)) + nx + cls.MARGIN
        size_z = int(round((max(shift['z'] for shift in shifts) - z0) / dz)) + nz + cls.MARGIN
        prior = matern_reference({**manifest, 'dims': [size_x, ny, size_z]}, anisotropy, *moments)
        state = cap_rh(clamp_log_rh(prior.sample(1, seed)), RH_CAP_OHMM)
        return cls((x0, z0), (dx, dz), state[:, 0].reshape(2, size_x, size_z), seed)

    @classmethod
    def load(cls, path):
        with np.load(path) as saved:
            return cls(saved['origin'], saved['spacing'], saved['fields'], saved['seed'])

    def save(self, path, anisotropy):
        """Fields plus the truth in the layout of the benchmark section (x, z centres in ft; Rh, Rv as
        (z, x) arrays in ohm m), which is what rh_rmse and compare-runs.py read."""
        x, z, rh = self.reference()
        log_rv = log_rv_from_latent(self.fields[0], self.fields[1], sequential.min_ratio, sequential.max_ratio,
                                    sequential.log_rv_cap, anisotropy)
        np.savez(path, origin=self.origin, spacing=self.spacing, fields=self.fields, seed=self.seed,
                 x=x, z=z, rh=rh, rv=np.exp(log_rv).T)

    def reference(self):
        centres = [self.origin[axis] + (np.arange(self.fields.shape[1 + axis]) + .5) * self.spacing[axis]
                   for axis in (0, 1)]
        return centres[0], centres[1], np.exp(self.fields[0]).T

    def window(self, shift, nx, nz):
        """(2, nx, nz) ln Rh and latent truth of the window at ``shift``."""
        offset = [(shift[key] - self.origin[axis]) / self.spacing[axis] for axis, key in enumerate('xz')]
        i, k = (int(round(value)) for value in offset)
        if max(abs(offset[0] - i), abs(offset[1] - k)) > 1e-6:
            raise ValueError('Window is not on the lattice of the twin truth')
        if i < 0 or k < 0 or i + nx > self.fields.shape[1] or k + nz > self.fields.shape[2]:
            raise ValueError('Window leaves the twin truth grid (it covers the --start/--stop range the '
                             'chain was started with)')
        return self.fields[:, i:i + nx, k:k + nz]


def carry_posterior(prior, state, nx, nz, movement, seed, rho=1., relaxation_rng=None, rho_mean=None):
    """Previous posterior as the next prior: reindex, relax toward the prior, fill entered cells."""
    entered = entered_cells(nx, nz, movement)
    shifted = reindex_window(state, nx, nz, movement)
    if np.any(np.asarray(rho) < 1.) or (rho_mean is not None and np.any(np.asarray(rho_mean) < 1.)):
        shifted = relax_to_prior(prior, shifted, rho, relaxation_rng, rho_mean)
    shifted = prior.condition_entered(shifted, entered.ravel(), seed)
    return cap_rh(clamp_log_rh(shifted), RH_CAP_OHMM), int(entered.sum())


def resume_point(output, rows, members):
    """Last checkpoint's posterior ensemble and index, plus the history and study so far."""
    indices = sorted(int(path.stem.rsplit('_', 1)[1])
                     for path in output.glob('inversion_results_assim_*.pkl'))
    if not indices:
        raise FileNotFoundError(f'No checkpoint to resume from in {output}')
    last = indices[-1]
    with (output / f'inversion_results_assim_{last}.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    state = np.asarray(saved['post_mda_params'], dtype=float)
    if state.shape != (rows, members) or not np.isfinite(state).all():
        raise ValueError('Checkpoint ensemble does not match this chain configuration')
    history_path = output / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    return state.copy(), last, history, json.loads((output / 'study.json').read_text())


DIRECTIONAL_COMPONENTS = ('USDP', 'USDA')


def directional_floor(covariance, floors):
    """Floor the standard deviation of the USDP / USDA channels at ``floors`` (deg, dB).

    Replaces the global sub-cell mask of these directional channels: where their
    signal-scaled sigma collapses (near-zero data in the early conductive stretch)
    the floor bounds the weight of sub-cell artefacts, elsewhere the channels keep
    their own sigma and the up/down information they carry.
    """
    covariance = np.array(covariance, dtype=float, copy=True)
    width = len(UDAR_COMPONENTS)
    if covariance.size % width or len(floors) != len(DIRECTIONAL_COMPONENTS) or min(floors) <= 0:
        raise ValueError('Need whole tools and one positive floor per directional component')
    for name, floor in zip(DIRECTIONAL_COMPONENTS, floors):
        positions = np.arange(UDAR_COMPONENTS.index(name), covariance.size, width)
        covariance[positions] = np.maximum(covariance[positions], float(floor) ** 2)
    return covariance


def model_error_factors(path, tool_pairs):
    """Per-channel model-error factors (1D stage, 2D stage) from a json file, in data-vector order.

    The file holds ``tools`` ([frequency, spacing] pairs), ``components`` and the tables ``factor_1d`` and
    ``factor_2d`` ([tool][component]): the standard deviation of the model error in units of the measurement
    standard deviation. A stage assimilates its data with the variance Cd (1 + factor^2).
    """
    table = json.loads(Path(path).read_text())
    tools = [tuple(tool) for tool in table['tools']]
    if list(table['components']) != list(UDAR_COMPONENTS) or set(tools) != {tuple(tool) for tool in tool_pairs}:
        raise ValueError('The model-error table does not match the tools and components of the data')
    out = []
    for key in ('factor_1d', 'factor_2d'):
        values = np.asarray(table[key], dtype=float)
        if values.shape != (len(tools), len(UDAR_COMPONENTS)) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f'{key} must be a finite non-negative table [tool][component]')
        out.append(np.concatenate([values[tools.index(tuple(tool))] for tool in tool_pairs]))
    return out


def fork_chain(source, destination, last_index, note, first_index=0):
    """New chain directory holding a source chain's state through ``last_index``.

    Copies the study (recording the fork), checkpoints first_index..last_index
    (``--resume`` needs only the last), the history entries and any gate records
    up to that index, so ``--resume`` continues at ``last_index + 1``.
    """
    import shutil

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    checkpoints = [source / f'inversion_results_assim_{index}.pkl' for index in range(first_index, last_index + 1)]
    missing = [path.name for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Source chain lacks {missing[:3]}')
    destination.mkdir(parents=True)
    study = json.loads((source / 'study.json').read_text())
    study['resumes'] = [entry for entry in study.get('resumes', []) if entry['from_index'] <= last_index]
    study['forked_from'] = {'chain': str(source), 'last_index': last_index, 'note': note}
    (destination / 'study.json').write_text(json.dumps(study, indent=2) + '\n')
    for path in checkpoints:
        shutil.copy2(path, destination / path.name)
    for pattern in ('gate-*.json', 'laplace-problem-*.npz', 'laplace-*.json'):
        for path in source.glob(pattern):
            # gate-031.json, laplace-problem-031.npz, laplace-problem-031-revert.npz, laplace-031.json
            if int(re.search(r'-(\d+)(?:-[a-z]+)?$', path.stem).group(1)) <= last_index:
                shutil.copy2(path, destination / path.name)
    history = [entry for entry in json.loads((source / 'history.json').read_text())
               if entry['logging_index'] <= last_index]
    (destination / 'history.json').write_text(json.dumps(history, indent=1) + '\n')
    if (source / 'tool.inp').exists():
        shutil.copy2(source / 'tool.inp', destination / 'tool.inp')
    return destination


def lateral_weights(nx, dx, mode, column_halfwidth=2, footprint_ft=40., taper_ft=60.):
    """Per-column weights of a 1D MDA increment along the well (bit column = nx // 2).

    'column': 1 within +- column_halfwidth columns of the bit, 0 beyond.
    'footprint': 1 to +- footprint_ft, cosine taper to 0 at +- taper_ft.
    """
    columns = np.abs(np.arange(nx) - nx // 2)
    if mode in ('full', 'ridge'):
        return np.ones(nx)
    if mode == 'column':
        return (columns <= column_halfwidth).astype(float)
    if mode == 'footprint':
        offset = columns * dx
        weights = (offset <= footprint_ft).astype(float)
        ramp = (offset > footprint_ft) & (offset < taper_ft)
        weights[ramp] = .5 * (1. + np.cos(np.pi * (offset[ramp] - footprint_ft) / (taper_ft - footprint_ft)))
        return weights
    raise ValueError(f'Unknown lateral update {mode}')


def pet_version():
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version('PET')
    except PackageNotFoundError:
        return None


def bit_column_indices(nx, nz):
    """State indices of the bit column (nx // 2), ln Rh block then latent block."""
    cells, bit = nx * nz, nx // 2
    return np.concatenate([block * cells + bit * nz + np.arange(nz) for block in (0, 1)])


def ridge_propagator(ensemble, nx, nz, radius=2):
    """Column-to-window regression of an ensemble's ridge (sparse-precision) estimate, no floors."""
    prior = EnsemblePrecisionPrior.fit(ensemble, (nx, nz), radius=radius, floors=(0., 0.))
    column = bit_column_indices(nx, nz)
    rest, propagate = conditional_regression(prior.precision(), column)
    return column, rest, propagate


def lateral_update(previous, updated, mode, weights, nx, nz, propagator=None):
    """Restrict ('column'), tile ('footprint') or regress ('ridge') a 1D MDA increment along the well.

    The 1D forward sees only the bit column, whose increment is the same in
    every mode. 'column' keeps the ensemble increment near the bit and drops
    it elsewhere; 'footprint' replaces it by each member's bit-column
    increment times the weight, so the update is coherent over the tool span;
    'ridge' replaces the rest of the window's increment by the regression of
    each member's column increment under the ensemble's ridge precision
    (``ridge_propagator``), instead of the Monte Carlo cross-covariance.
    """
    if mode == 'full':
        return updated
    if mode == 'ridge':
        column, rest, propagate = propagator
        increment = np.empty_like(previous)
        increment[column] = (updated - previous)[column]
        increment[rest] = propagate(increment[column])
        return cap_rh(clamp_log_rh(previous + increment), RH_CAP_OHMM)
    increment = (updated - previous).reshape(2, nx, nz, -1)
    if mode == 'column':
        increment = increment * weights[None, :, None, None]
    else:
        increment = weights[None, :, None, None] * increment[:, nx // 2:nx // 2 + 1]
    return cap_rh(clamp_log_rh(previous + increment.reshape(previous.shape)), RH_CAP_OHMM)


def spread_summary(state, nx, nz):
    cells = nx * nz
    std = state[:cells].std(axis=1, ddof=1).reshape(nx, nz)
    return {'mean_std_ln_rh_window': float(std.mean()),
            'mean_std_ln_rh_bit_column': float(std[nx // 2].mean())}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE,
                        help='Sequential run whose run.json supplies the window, data and prior moments')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=31, help='Exclusive end index')
    parser.add_argument('--ensemble-size', type=int, default=250)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument('--mda-steps', type=int, default=10)
    parser.add_argument('--tempering', choices=['linear-inflation', 'equal-energy'],
                        default='linear-inflation')
    parser.add_argument('--directional-sigma-floor', type=float, nargs=2, default=None, metavar=('USDP', 'USDA'),
                        help='Use the USDP/USDA channels (no global sub-cell mask) with their sigma floored '
                             'at these values (deg, dB)')
    parser.add_argument('--likelihood-weight', type=float, default=1.,
                        help='Fraction of the data energy assimilated per point: the MDA inflations '
                             'satisfy sum(1/alpha) = weight (equal-energy: alpha = steps / weight)')
    parser.add_argument('--anisotropy', choices=['probit', 'log_clipped'], default='log_clipped')
    parser.add_argument('--anisotropy-mean', type=float, default=.5)
    parser.add_argument('--anisotropy-std', type=float, default=.5)
    parser.add_argument('--memory-ft', type=float, default=None,
                        help='Relax the carried ensemble toward the Matern prior between points with '
                             'rho = exp(-dMD / memory); omit for no relaxation')
    parser.add_argument('--memory-mean-factor', type=float, default=4.,
                        help='Multiply the memory length of the ensemble MEAN by this factor: the carried '
                             'local mean (e.g. a resolved transition) is pulled back to the prior mean '
                             'more slowly than the spread relaxes (>1 keeps the mean; 1 = same memory). '
                             'Only used with --memory-ft')
    parser.add_argument('--model-error', type=Path, default=None,
                        help='json table of per-channel model-error factors (see model_error_factors): the 1D '
                             'update and the exported 2D problem assimilate the data with Cd (1 + factor^2) of '
                             'their stage. Misfits and the gate statistic keep the measurement Cd')
    parser.add_argument('--gate-covariance', choices=['measurement', 'stage'], default='measurement',
                        help="Covariance of the gate statistic: 'measurement' (Cd) or 'stage' (the 1D stage's "
                             'Cd (1 + factor^2) of --model-error, so the gate opens only when the 1D misfit '
                             'exceeds what the known 1D model error explains)')
    parser.add_argument('--twin-seed', type=int, default=None,
                        help='Twin experiment: replace the measured data by the 1D response of a synthetic '
                             'truth drawn from the Matern prior with this seed, plus noise from the data '
                             'covariance. Calibration is then a test of the update scheme alone. The truth is '
                             'written to twin_truth.npz; a resumed chain reuses it (flag not needed)')
    parser.add_argument('--gate-threshold', type=float, default=None,
                        help='Open the 2D gate when the post-1D member-median chi2/n exceeds this; '
                             'the 1D update of that point is discarded (probe-and-revert)')
    parser.add_argument('--on-gate', choices=['stop', 'continue'], default='stop',
                        help='stop: export the gate point for the Laplace 2D stage and end the chain; '
                             'continue: carry the reverted prior to the next point')
    parser.add_argument('--cascade-threshold', type=float, default=None,
                        help='Cascade: after the --likelihood-weight share of the data energy (1D MDA), '
                             "send the point to the 2D Laplace stage for the rest when the best member's 1D "
                             'chi2/n exceeds this, keeping the 1D update; otherwise assimilate the rest with '
                             'the same number of 1D MDA steps')
    parser.add_argument('--compare-revert', choices=['never', 'first'], default='never',
                        help='Cascade: at the first 2D point also export the reverted problem (pre-1D '
                             'ensemble, full energy) for a side-by-side check')
    parser.add_argument('--lateral-update', choices=['full', 'column', 'footprint', 'ridge'], default='full',
                        help="Along-well extent of every 1D MDA increment: 'full' (ensemble update of the whole "
                             "window), 'column' (only the bit column +- --column-halfwidth columns), "
                             "'footprint' (each member's bit-column increment copied along the well, full "
                             'weight to +- --footprint-ft, cosine-tapered to zero at --footprint-taper-ft) or '
                             "'ridge' (the column's increment regressed onto the window with the point's "
                             'ridge-regression sparse precision instead of the Monte Carlo cross-covariance)')
    parser.add_argument('--analysis', choices=['esmda', 'enif'], default='esmda',
                        help="1D MDA analysis: 'esmda' (ensemble Kalman update, PET's approx flavour) or "
                             "'enif' (PET's EnIF-MDA, seeded with the Matern prior precision; needs PET >= 0.3)")
    parser.add_argument('--student-weights', choices=['mean', 'members'], default=None,
                        help='Heavy-tailed (Student-t) increment prior in the 1D EnIF update by reweighting: '
                             'after the usual update with the Matern precision, the point is updated again from '
                             'its carried prior with the precision of a Gaussian increment prior whose face '
                             "weights are the Student-t weights at the previous round's posterior, evaluated "
                             "at the increments of its ensemble mean ('mean') or at the member average of "
                             "the squared increments ('members'). Needs --analysis enif")
    parser.add_argument('--student-rounds', type=int, default=1,
                        help='Reweighted updates per point after the Matern round')
    parser.add_argument('--student-dof', type=float, default=1.)
    parser.add_argument('--student-scales', type=float, nargs=4, default=(.1, .03, .05, .015),
                        metavar=('RH_VERTICAL', 'RH_LATERAL', 'LATENT_VERTICAL', 'LATENT_LATERAL'),
                        help='Student-t scales of the increments between neighbouring cells (state units)')
    parser.add_argument('--student-anchor', type=float, nargs=2, default=(3., 1.5), metavar=('RH', 'LATENT'),
                        help='Std of the weak cellwise anchor that makes the increment precision proper')
    parser.add_argument('--merge-behind', type=int, nargs='+', default=None, metavar='POINTS',
                        help='Merged logging points in the 2D stage: also assimilate the data of the points this '
                             'many logging indices behind the gate point (e.g. 5 10), with the tool simulated '
                             'at their positions inside the current window')
    parser.add_argument('--merge-own-history', action='store_true',
                        help='Merge only once every --merge-behind point is a point this chain has assimilated '
                             '(at or after its first logging point). Before that the 2D stage uses the gate '
                             'point alone with its full share, instead of data from before the chain began, '
                             'in a part of the window it has never updated')
    parser.add_argument('--merge-shares', type=float, nargs='+', default=None, metavar='SHARE',
                        help='Share of the data energy of the current point and of each --merge-behind point '
                             '(covariance Cd / share). A datum is used once as the current point and once per '
                             'offset, so shares that sum to one use it exactly once in total '
                             '(default: 1/2 for the current point, the rest split evenly)')
    parser.add_argument('--two-d-prior', choices=['matern_cellwise', 'ensemble_precision'],
                        default='matern_cellwise',
                        help='2D Laplace prior at probe-and-revert gate points, built from the carried ensemble')
    parser.add_argument('--column-halfwidth', type=int, default=2)
    parser.add_argument('--footprint-ft', type=float, default=40.)
    parser.add_argument('--footprint-taper-ft', type=float, default=60.)
    parser.add_argument('--resume', action='store_true',
                        help='Continue an existing chain from its last checkpoint (e.g. after the '
                             'Laplace stage wrote the gate point)')
    args = parser.parse_args(argv)
    if args.resume and not (args.output_dir / 'study.json').exists():
        parser.error('--resume needs an existing chain directory')
    if not 0 <= args.start < args.stop:
        parser.error('Require 0 <= start < stop')
    if args.memory_ft is not None and not (np.isfinite(args.memory_ft) and args.memory_ft > 0):
        parser.error('Require a positive finite memory length')
    if not args.memory_mean_factor > 0:
        parser.error('Require a positive mean memory factor')
    if args.twin_seed is not None and (args.gate_threshold is not None or args.cascade_threshold is not None):
        parser.error('--twin-seed is a 1D-only experiment: omit --gate-threshold and --cascade-threshold')
    if args.gate_threshold is not None and not (np.isfinite(args.gate_threshold) and args.gate_threshold > 0):
        parser.error('Require a positive finite gate threshold')
    if args.ensemble_size < 2 or not 1 <= args.mda_steps <= 10:
        parser.error('Require ensemble-size >= 2 and 1 <= mda-steps <= 10')
    if not (np.isfinite(args.likelihood_weight) and 0 < args.likelihood_weight <= 1):
        parser.error('Require 0 < likelihood-weight <= 1')
    if args.cascade_threshold is not None:
        if not (np.isfinite(args.cascade_threshold) and args.cascade_threshold > 0):
            parser.error('Require a positive finite cascade threshold')
        if args.likelihood_weight >= 1:
            parser.error('The cascade needs a first share --likelihood-weight below one')
        if args.gate_threshold is not None or args.on_gate != 'stop':
            parser.error('The cascade replaces the probe-and-revert gate (no --gate-threshold, --on-gate stop)')
    elif args.compare_revert != 'never':
        parser.error('--compare-revert needs --cascade-threshold')
    if args.analysis == 'enif' and (args.lateral_update != 'full' or args.cascade_threshold is not None):
        parser.error('EnIF-MDA updates the whole window itself: use it with --lateral-update full and no cascade')
    if args.merge_behind:
        if args.merge_shares is None:
            args.merge_shares = [.5] + [.5 / len(args.merge_behind)] * len(args.merge_behind)
        if (min(args.merge_behind) < 1 or len(set(args.merge_behind)) != len(args.merge_behind)
                or len(args.merge_shares) != len(args.merge_behind) + 1
                or not all(0 < share <= 1 for share in args.merge_shares)):
            parser.error('Require distinct positive --merge-behind offsets and one share in (0, 1] for the '
                         'current point and for each offset')
        if args.cascade_threshold is not None or args.twin_seed is not None or args.gate_threshold is None:
            parser.error('--merge-behind acts on the 2D stage of a gated chain (no cascade, no twin)')
    elif args.merge_shares is not None or args.merge_own_history:
        parser.error('--merge-shares and --merge-own-history need --merge-behind')
    if args.student_weights is not None and args.analysis != 'enif':
        parser.error('--student-weights reweights the EnIF precision: use it with --analysis enif')
    if (args.student_rounds < 1 or not args.student_dof > 0 or min(args.student_scales) <= 0
            or min(args.student_anchor) <= 0):
        parser.error('Require student-rounds >= 1 and positive Student-t dof, scales and anchor')
    if args.column_halfwidth < 0 or not 0 <= args.footprint_ft < args.footprint_taper_ft:
        parser.error('Require column-halfwidth >= 0 and 0 <= footprint-ft < footprint-taper-ft')
    if not np.isfinite([args.anisotropy_mean, args.anisotropy_std]).all() or args.anisotropy_std <= 0:
        parser.error('Anisotropy moments must be finite with positive std')
    return args


def main(argv=None):
    args = parse_args(argv)
    reference = args.reference.resolve()
    if args.resume:
        started = json.loads((args.output_dir / 'study.json').read_text())['reference']
        if Path(started).resolve() != reference:
            raise ValueError(f'Resume with reference {reference}, but the chain was started from {started} '
                             '(data and window come from its run.json)')
    manifest = json.loads((reference / 'run.json').read_text())
    benchmark = Path(manifest['benchmark'])
    nx, ny, nz = manifest['dims']
    dx, dy, dz = manifest['spacing_ft']
    cells, bit = nx * nz, nx // 2
    data = pd.read_pickle(ROOT / manifest['data'])
    variances = pd.read_pickle(ROOT / manifest['variance'])
    tool_pairs = list(data.columns)
    if not args.stop <= len(data):
        raise ValueError('Assimilation range lies outside the data')
    model_error = None if args.model_error is None else model_error_factors(args.model_error, tool_pairs)
    sequential.dims = (nx, ny, nz)
    sequential.dX, sequential.dY, sequential.dZ = dx, dy, dz
    sequential.tools = [str(tool) for tool in tool_pairs]
    sequential.max_ratio = float(manifest['prior_ratio_max'])
    sequential.log_rv_cap = float(np.log(RV_CAP_OHMM))
    sequential.anisotropy = args.anisotropy
    prior = matern_reference(manifest, args.anisotropy, args.anisotropy_mean, args.anisotropy_std)
    matern_precision = prior.precision() if args.analysis == 'enif' else None
    inflation = cascade_inflation(args.mda_steps, args.tempering, args.likelihood_weight)
    weights = lateral_weights(nx, dx, args.lateral_update, args.column_halfwidth, args.footprint_ft,
                              args.footprint_taper_ft)
    lateral = {'mode': args.lateral_update, 'column_halfwidth': args.column_halfwidth,
               'footprint_ft': args.footprint_ft, 'footprint_taper_ft': args.footprint_taper_ft,
               'weights': weights.tolist()}
    if args.lateral_update == 'ridge':
        lateral['ridge'] = {'radius': PROPAGATION_RADIUS, 'floors': [0., 0.], 'fit': 'once per point, on the '
                            'carried ensemble before its 1D update (the rest-on-column regression is invariant '
                            'under data that see only the column)'}
    cascading = args.cascade_threshold is not None
    completion = (cascade_inflation(args.mda_steps, args.tempering, 1. - args.likelihood_weight)
                  if cascading else np.zeros(0))
    truth = reference_truth(benchmark)
    window = LocalWindow((nx, ny, nz), (dx, dy, dz))
    output = args.output_dir.resolve()
    seeds = np.random.SeedSequence(args.seed).spawn(args.stop)
    study = {'method': 'sequential_matern_1d_esmda', 'reference': str(reference),
             'benchmark': str(benchmark), 'start': args.start, 'stop': args.stop,
             'ensemble_size': args.ensemble_size, 'seed': args.seed,
             'mda_steps': args.mda_steps, 'tempering': args.tempering,
             'covariance_inflation': inflation.tolist(), 'likelihood_weight': args.likelihood_weight,
             'directional_sigma_floor': args.directional_sigma_floor,
             'model_error': ('none' if args.model_error is None else
                             {'file': str(args.model_error.resolve()),
                              'rule': 'stage variance = Cd (1 + factor^2) per channel; misfits and the gate '
                                      'statistic use the measurement Cd'}),
             'twin': ('none' if args.twin_seed is None else
                      {'seed': args.twin_seed, 'truth': 'twin_truth.npz',
                       'rule': 'data = 1D response of the bit column of a Matern prior draw + N(0, Cd) noise; '
                               'channel mask and Cd are those of the measured data'}),
             'fidelity': '1D only (no 0D, no 2D, no misfit gate)', 'localization': 'none',
             'prior': {'family': 'matern', 'nu': 1., 'boundary': 'robin',
                       'rh_ranges_ft': [RH_CORRELATION_X_FT, RH_CORRELATION_Z_FT],
                       'latent_ranges_ft': [manifest['prior_latent_correlation_x_ft'],
                                            manifest['prior_latent_correlation_z_ft']],
                       'means': [float(prior.blocks[0].mean[0]), float(prior.blocks[1].mean[0])],
                       'stds': [float(prior.blocks[0].std[0]), float(prior.blocks[1].std[0])]},
             'window_carry': 'reindex kept cells; entered cells redrawn from the Matern GMRF '
                             'conditionally on the kept cells (prior.condition_entered)',
             'relaxation': ('none' if args.memory_ft is None else
                            {'memory_ft': args.memory_ft,
                             'rule': 'x <- mu + rho_mean (m - mu) + rho (x - m) + sqrt(1 - rho^2) L eps, m the '
                                     'ensemble mean, rho = exp(-dMD / memory), rho_mean = exp(-dMD / (memory '
                                     '* mean_factor)), before the entered-cell fill',
                             'adaptive': {'mean_factor': args.memory_mean_factor}}),
             'anisotropy': {'parameterization': args.anisotropy,
                            'state_moments': [args.anisotropy_mean, args.anisotropy_std]},
             'lateral_update': lateral, 'two_d_prior': args.two_d_prior,
             'analysis': ('esmda' if args.analysis == 'esmda' else
                          {'method': 'enif', 'pet_version': pet_version(),
                           'precision': 'Matern prior precision (SPDE), fed at the first MDA step of every '
                                        'point; PET carries the posterior precision across the steps',
                           'regression': 'PET/graphite-maps boosted linear regression of the responses on '
                                         'the standardised window state, refitted every step'}),
             'merge': ('none' if not args.merge_behind else
                       {'behind': list(args.merge_behind), 'shares': list(args.merge_shares),
                        'own_history': bool(args.merge_own_history),
                        'rule': '2D stage: data of the gate point and of the points this many indices behind '
                                'it, each with its 2D covariance / share; tool positions simulated inside the '
                                'window of the gate point. 1D stage: the current point only'}),
             'increment_prior': ('none' if args.student_weights is None else
                                 {'family': 'student-t increments, by reweighting', 'weights': args.student_weights,
                                  'rounds': args.student_rounds, 'dof': args.student_dof,
                                  'scales': list(args.student_scales), 'anchor': list(args.student_anchor),
                                  'rule': 'round 0: EnIF-MDA with the Matern precision. Round k: the update is '
                                          'repeated from the carried prior with the precision Dz^T wz Dz + '
                                          'Dx^T wx Dx + diag(1 / anchor^2), w = (dof + 1) / (dof scale^2 + d2), '
                                          'd2 the squared increment of the posterior of round k - 1 (of its '
                                          'mean, or averaged over its members)'}),
             'cascade': ('none' if not cascading else
                         {'threshold': args.cascade_threshold,
                          'statistic': "best member's 1D chi2/n (original Cd) after the first share",
                          'first_share': args.likelihood_weight,
                          'completion_inflation': completion.tolist(),
                          'two_d_share': 1. - args.likelihood_weight,
                          'two_d_prior': 'ensemble_precision (local ridge modified Cholesky of the '
                                         'post-1D ensemble)',
                          'two_d_start': 'ensemble mean', 'compare_revert': args.compare_revert,
                          'rule': 'keep the 1D update; 2D Laplace for the remaining share when the best '
                                  'member exceeds the threshold, else finish with 1D MDA'}),
             'gate': ('none' if args.gate_threshold is None else
                      {'statistic': 'post-1D member-median reduced chi2 against the measurement Cd'
                                    if args.gate_covariance == 'measurement' else
                                    'post-1D member-median reduced chi2 against the 1D stage covariance '
                                    'Cd (1 + factor^2)', 'covariance': args.gate_covariance,
                       'threshold': args.gate_threshold, 'on_open': args.on_gate,
                       'rule': 'probe-and-revert: the 1D update is discarded when the gate opens'}),
             'rh_limits_ln': [-1., 4.], 'rh_cap_ohmm': RH_CAP_OHMM, 'rv_cap_ohmm': RV_CAP_OHMM,
             'dims': [nx, ny, nz], 'spacing_ft': [dx, dy, dz], 'tools': [list(tool) for tool in tool_pairs]}
    start, state, previous_index, history = args.start, None, None, []
    if args.resume:
        state, previous_index, history, existing = resume_point(output, 2 * cells, args.ensemble_size)
        if previous_index + 1 >= args.stop:
            raise ValueError(f'Chain already reached index {previous_index}; raise --stop')
        start = previous_index + 1
        existing.setdefault('resumes', []).append(
            {'from_index': previous_index, 'stop': args.stop, 'memory_ft': args.memory_ft,
             'memory_mean_factor': args.memory_mean_factor,
             'gate_threshold': args.gate_threshold, 'on_gate': args.on_gate,
             'cascade_threshold': args.cascade_threshold, 'compare_revert': args.compare_revert,
             'lateral_update': lateral, 'directional_sigma_floor': args.directional_sigma_floor})
        if args.directional_sigma_floor != existing.get('directional_sigma_floor'):
            existing['directional_sigma_floor_from_index'] = previous_index + 1
        existing['directional_sigma_floor'] = args.directional_sigma_floor
        existing['stop'] = args.stop
        (output / 'study.json').write_text(json.dumps(existing, indent=2) + '\n')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'study.json').write_text(json.dumps(study, indent=2) + '\n')

    def point_data(point):
        """Retained data of a logging point: observations, measurement Cd, channel mask, and the
        covariances the 1D and 2D stages assimilate with (model error included)."""
        observations, Cd = observation_vectors(data, variances, point, tool_pairs)
        sub_cell = np.zeros(len(observations), dtype=bool)
        if args.directional_sigma_floor is None:
            sub_cell[sequential.component_positions(len(observations))] = True
        else:
            Cd = directional_floor(Cd, args.directional_sigma_floor)
        mask = ~(noise_floor_mask(observations, Cd) | sub_cell)
        if not mask.any():
            raise ValueError(f'Index {point}: every channel is masked')
        observations, Cd = observations[mask], Cd[mask]
        # Cd: measurement covariance (misfits, gate). Cd_1d / Cd_2d: what each stage assimilates with.
        Cd_1d, Cd_2d = Cd, Cd
        if model_error is not None:
            Cd_1d, Cd_2d = (Cd * (1. + factor[mask] ** 2) for factor in model_error)
        return observations, Cd, mask, Cd_1d, Cd_2d

    previous_cwd = Path.cwd()
    try:
        os.chdir(output)
        sequential.setup_simulators(benchmark, sequential.tools)
        twin = None
        if args.resume and (output / 'twin_truth.npz').exists():
            twin = TwinTruth.load(output / 'twin_truth.npz')
        elif args.twin_seed is not None:
            if args.resume:
                raise ValueError('This chain was not started as a twin experiment')
            replay = LocalWindow((nx, ny, nz), (dx, dy, dz))
            shifts = [replay.update((sequential.WX[i], sequential.WY[i], sequential.TVD[i]))[0]
                      for i in range(args.stop)][args.start:]
            twin = TwinTruth.draw(manifest, shifts, args.anisotropy,
                                  (args.anisotropy_mean, args.anisotropy_std), args.twin_seed)
            twin.save(output / 'twin_truth.npz', args.anisotropy)
        if twin is not None:
            truth = (*twin.reference(), None)
        for index in range(args.stop):
            # Advance the window from the trajectory start so the origin matches the reference.
            current_shift, movement = window.update(
                (sequential.WX[index], sequential.WY[index], sequential.TVD[index]))
            if index < start:
                continue
            started = time.monotonic()
            sequential.current_shift = current_shift
            fill_seed, perturbation_seed, relaxation_seed = seeds[index].spawn(3)
            rng = np.random.default_rng(perturbation_seed)
            observations, Cd, mask, Cd_1d, Cd_2d = point_data(index)
            if twin is not None:
                column = twin.window(current_shift, nx, nz)[:, bit].reshape(2 * nz)
                observations = (sequential.simulate_pred_only(column, index, '1D')[mask]
                                + np.sqrt(Cd) * np.random.default_rng([twin.seed, index]).normal(size=Cd.shape))
            phase, phase_wraps = phase_channels(mask, UDAR_COMPONENTS), [0]

            def predict_1d(fields):
                # Predicted phases on the observed branch: innovations, misfits and the gate
                # statistic never see a 360-degree jump.
                columns = fields.reshape(2, nx, nz, -1)[:, bit].reshape(2 * nz, -1)
                raw = np.column_stack([
                    sequential.simulate_pred_only(columns[:, member], index, '1D')[mask]
                    for member in range(columns.shape[1])])
                aligned = align_phases(raw, observations, phase)
                phase_wraps[0] += int(np.sum(aligned != raw))
                return aligned

            step_ft, rho, rho_mean = None, 1., None
            if state is None:
                state = cap_rh(clamp_log_rh(prior.sample(args.ensemble_size, fill_seed)), RH_CAP_OHMM)
                entered = cells
            else:
                step_ft = float(sequential.MD[index] - sequential.MD[previous_index])
                if args.memory_ft is not None:
                    rho = float(np.exp(-abs(step_ft) / args.memory_ft))
                    if args.memory_mean_factor != 1:
                        rho_mean = float(np.exp(-abs(step_ft) / (args.memory_ft * args.memory_mean_factor)))
                state, entered = carry_posterior(prior, state, nx, nz, movement, fill_seed, rho,
                                                 np.random.default_rng(relaxation_seed), rho_mean)
            point_prior = state.copy()
            meta = {'dims': (nx, ny, nz), 'spacing_ft': (dx, dy, dz), 'shift_ft': dict(current_shift)}
            propagator = (ridge_propagator(point_prior, nx, nz, PROPAGATION_RADIUS)
                          if args.lateral_update == 'ridge' else None)
            prior_predicted = predict_1d(state)
            reweighting, matern_round = [], None
            for update_round in range(1 if args.student_weights is None else 1 + args.student_rounds):
                precision = matern_precision
                if update_round:
                    # Student-t increments by reweighting: face weights at the previous round's posterior,
                    # then the same update again from the carried prior.
                    face_weights = ensemble_student_weights(state, (nx, nz), args.student_dof,
                                                            args.student_scales, args.student_weights)
                    precision = increment_precision((nx, nz), *face_weights, args.student_anchor)
                    ceiling = (args.student_dof + 1.) / (args.student_dof * np.asarray(
                        args.student_scales).reshape(2, 2)[0] ** 2)
                    reweighting.append({'round': update_round, 'rh_rmse_log10_before': steps[-1]['rh_rmse_log10'],
                                        'member_1d_misfit_median_before': steps[-1]['member_1d_misfit_median'],
                                        'rh_vertical_weight_median': float(np.median(face_weights[0][0]) / ceiling[0]),
                                        'rh_vertical_faces_below_tenth': float(np.mean(
                                            face_weights[0][0] < .1 * ceiling[0])),
                                        'rh_lateral_faces_below_tenth': float(np.mean(
                                            face_weights[1][0] < .1 * ceiling[1]))})
                    if update_round == 1:
                        # The gate asks whether a 1D model can fit the point: answered by the Matern round,
                        # so chains with and without reweighting send the same kind of point to 2D.
                        matern_round = (steps, predicted)
                    state = point_prior.copy()
                enif = (EnIFProbe(precision, observations, Cd_1d, inflation, int(rng.integers(2 ** 31 - 1)))
                        if args.analysis == 'enif' else None)
                predicted = prior_predicted
                steps = [{'step': 0, 'alpha': None,
                          **member_misfit_summary(predicted, observations, Cd),
                          'rh_rmse_log10': rh_rmse(state, meta, truth)}]
                for step, alpha in enumerate(inflation, start=1):
                    if enif is not None:
                        perturbed = rng.normal(observations[:, None], np.sqrt(float(alpha) * Cd_1d)[:, None],
                                               predicted.shape)
                        state = cap_rh(clamp_log_rh(state + enif.step(state, predicted, perturbed)), RH_CAP_OHMM)
                    else:
                        state = lateral_update(state, cap_rh(update(state, predicted, observations, Cd_1d,
                                                                    float(alpha), rng, cells, None), RH_CAP_OHMM),
                                               args.lateral_update, weights, nx, nz, propagator)
                    predicted = predict_1d(state)
                    steps.append({'step': step, 'alpha': float(alpha),
                                  **member_misfit_summary(predicted, observations, Cd),
                                  'rh_rmse_log10': rh_rmse(state, meta, truth)})
            cascade = None
            if cascading:
                first = np.mean((predicted - observations[:, None]) ** 2 / Cd[:, None], axis=0)
                best = int(np.argmin(first))
                cascade = {'threshold': args.cascade_threshold, 'first_share': args.likelihood_weight,
                           'first_share_steps': len(inflation), 'best_member': best,
                           'best_member_misfit': float(first[best]),
                           'first_share_member_median': float(np.median(first))}
                cascade['decision'] = '2d' if cascade['best_member_misfit'] > args.cascade_threshold else '1d'
                if cascade['decision'] == '1d':
                    for step, alpha in enumerate(completion, start=len(inflation) + 1):
                        state = lateral_update(state, cap_rh(update(state, predicted, observations, Cd_1d,
                                                                    float(alpha), rng, cells, None),
                                                             RH_CAP_OHMM),
                                               args.lateral_update, weights, nx, nz, propagator)
                        predicted = predict_1d(state)
                        steps.append({'step': step, 'alpha': float(alpha),
                                      **member_misfit_summary(predicted, observations, Cd),
                                      'rh_rmse_log10': rh_rmse(state, meta, truth)})
            gate_steps, gate_predicted = (steps, predicted) if matern_round is None else matern_round
            probe = {'threshold': args.gate_threshold if cascade is None else args.cascade_threshold,
                     'pre_update_mean_prediction_misfit': gate_steps[0]['mean_prediction_1d_misfit'],
                     'post_member_median_misfit': gate_steps[-1]['member_1d_misfit_median'],
                     'post_mean_prediction_misfit': gate_steps[-1]['mean_prediction_1d_misfit']}
            # The same statistic against the covariance the 1D stage assimilated with (model error included).
            probe['post_member_median_misfit_stage'] = float(np.median(
                np.mean((gate_predicted - observations[:, None]) ** 2 / Cd_1d[:, None], axis=0)))
            if matern_round is not None:
                probe['update'] = 'Matern round (before the reweighted rounds)'
                probe['reweighted_post_member_median_misfit'] = steps[-1]['member_1d_misfit_median']
                probe['reweighted_post_member_median_misfit_stage'] = float(np.median(
                    np.mean((predicted - observations[:, None]) ** 2 / Cd_1d[:, None], axis=0)))
            probe['gate_covariance'] = args.gate_covariance
            if cascade is None:
                statistic = probe['post_member_median_misfit_stage' if args.gate_covariance == 'stage'
                                  else 'post_member_median_misfit']
                probe['open'] = bool(args.gate_threshold is not None and statistic > args.gate_threshold)
            else:
                probe.update(kind='cascade', best_member=cascade['best_member'],
                             best_member_misfit=cascade['best_member_misfit'],
                             open=cascade['decision'] == '2d', reverted=False)
            reverted = probe['open'] and cascade is None
            if reverted:
                # Probe-and-revert: the 1D operator cannot fit this point, so its update is
                # discarded and the carried (relaxed) prior is what the 2D stage starts from.
                state, predicted = point_prior.copy(), prior_predicted
            member_misfit = np.mean((predicted - observations[:, None]) ** 2 / Cd[:, None], axis=0)
            prior_diagnostics = {**ensemble_diagnostics(point_prior, nx, nz, prior, args.anisotropy),
                                 **spread_summary(point_prior, nx, nz)}
            posterior_diagnostics = {**ensemble_diagnostics(state, nx, nz, prior, args.anisotropy),
                                     **spread_summary(state, nx, nz)}
            record = {'logging_index': index, 'previous_index': previous_index,
                      'window_movement': [int(m) for m in movement], 'entered_cells': int(entered),
                      'shift_ft': dict(current_shift), 'retained_channels': int(mask.sum()),
                      'along_well_step_ft': step_ft, 'relaxation_rho': rho, 'relaxation_rho_mean': rho_mean,
                      'gate': probe, 'reverted': reverted, 'cascade': cascade, 'phase_wraps': phase_wraps[0],
                      'reweighting': reweighting,
                      'lateral_update': args.lateral_update,
                      'prior': {**steps[0], **prior_diagnostics},
                      'posterior': {**(steps[0] if reverted else steps[-1]), **posterior_diagnostics},
                      'steps': steps, 'seconds': time.monotonic() - started}
            history.append(record)
            (output / 'history.json').write_text(json.dumps(history, indent=1) + '\n')
            if probe['open'] and args.on_gate == 'stop':
                problem = f'laplace-problem-{index:03d}.npz'
                merged = {}
                first_point = history[0]['logging_index']          # where this chain began (also after a resume)
                if args.merge_behind and not (args.merge_own_history
                                              and index - max(args.merge_behind) < first_point):
                    # Merged logging points: the 2D stage also simulates the tool at earlier points inside
                    # this window. Every point enters with its covariance divided by its share.
                    behind = [index - offset for offset in args.merge_behind if index - offset >= 0]
                    parts = [(index, args.merge_shares[0], observations, Cd_2d, mask)]
                    for point, share in zip(behind, args.merge_shares[1:]):
                        earlier = point_data(point)
                        parts.append((point, share, earlier[0], earlier[4], earlier[2]))
                    merged = dict(tool_indices=[part[0] for part in parts], merge_shares=[part[1] for part in parts],
                                  merge_channels=[int(part[4].sum()) for part in parts],
                                  measurement_variance=np.concatenate([part[3] for part in parts]))
                    merged_data = dict(observations=np.concatenate([part[2] for part in parts]),
                                       variance=np.concatenate([part[3] / part[1] for part in parts]),
                                       mask=np.concatenate([part[4] for part in parts]))
                else:
                    merged_data = dict(observations=observations, variance=Cd_2d, mask=mask)
                common = dict(index=index, **merged_data, **merged,
                              shift=current_shift, movement=movement, dims=(nx, ny, nz),
                              spacing=(dx, dy, dz), tools=tool_pairs, anisotropy=args.anisotropy,
                              anisotropy_moments=(args.anisotropy_mean, args.anisotropy_std),
                              ranges=[[RH_CORRELATION_X_FT, RH_CORRELATION_Z_FT],
                                      [manifest['prior_latent_correlation_x_ft'],
                                       manifest['prior_latent_correlation_z_ft']]],
                              well=sequential.wellpath)
                revert_problem = None
                if cascade is None:
                    export_problem(output / problem, state=point_prior, probe=probe,
                                   prior_type=args.two_d_prior, **common)
                else:
                    # The 2D stage starts from the 1D-updated ensemble and takes the remaining share.
                    first_gate = not any(output.glob('gate-*.json'))
                    export_problem(output / problem, state=state, probe=probe,
                                   likelihood_weight=1. - args.likelihood_weight,
                                   prior_type='ensemble_precision', **common)
                    if args.compare_revert == 'first' and first_gate:
                        revert_problem = f'laplace-problem-{index:03d}-revert.npz'
                        export_problem(output / revert_problem, state=point_prior,
                                       probe={**probe, 'reverted': True}, likelihood_weight=1.,
                                       prior_type='ensemble_precision', **common)
                (output / f'gate-{index:03d}.json').write_text(json.dumps(
                    {'logging_index': index, 'previous_index': previous_index, 'probe': probe,
                     'problem': problem, 'revert_problem': revert_problem, 'cascade': cascade,
                     'shift_ft': dict(current_shift),
                     'window_movement': [int(m) for m in movement], 'relaxation_rho': float(np.mean(rho)),
                     'entered_cells': int(entered)}, indent=1) + '\n')
                if cascade is None:
                    print(f"Index {index}: gate open (post-1D member median "
                          f"{probe['post_member_median_misfit']:.4g} > {args.gate_threshold:g}); "
                          f"1D update reverted, problem exported to {problem}; chain stopped", flush=True)
                else:
                    print(f"Index {index}: cascade to 2D (best member {cascade['best_member_misfit']:.4g} > "
                          f"{args.cascade_threshold:g} after {args.likelihood_weight:g} of the energy); "
                          f"problem exported to {problem}"
                          f"{f' and {revert_problem}' if revert_problem else ''}; chain stopped", flush=True)
                break
            misfit = {'1D_initial': steps[0]['mean_prediction_1d_misfit'],
                      '1D_final': steps[-1]['mean_prediction_1d_misfit'],
                      'before_final': steps[0]['mean_prediction_1d_misfit'],
                      'after_final': steps[-1]['mean_prediction_1d_misfit'],
                      'after_final_kind': 'forward'}
            results = {'prior_params': point_prior, 'post_mda_params': state.copy(),
                       'post_1d_params': state.copy(),
                       'posterior_ensemble_params': state.T.copy(),
                       'posterior_params': state.T.copy(), 'posterior_predictions': predicted.T.copy(),
                       'posterior_misfit': member_misfit.tolist(), 'observations': observations,
                       'data_covariance': Cd, 'update_covariance': Cd_1d, 'observation_mask': mask,
                       'masked_channels': int((~mask).sum()), 'dims': (nx, ny, nz),
                       'spacing_ft': (dx, dy, dz), 'shift_ft': dict(current_shift),
                       'window_movement': movement, 'entered_cells': int(entered),
                       'previous_index': previous_index, 'logging_index': index,
                       'tools': tool_pairs, 'seed': args.seed, 'misfit': misfit,
                       'misfit_threshold': None,
                       'mda_weight': args.likelihood_weight if cascade is None else 1.,
                       'lower_fidelity_weight': 1.,
                       'covariance_inflation': [step['alpha'] for step in steps[1:]], 'mda_history': steps,
                       'cascade': cascade,
                       'memory_ft': args.memory_ft, 'relaxation_rho': rho, 'relaxation_rho_mean': rho_mean,
                       'gate': probe, 'reverted': reverted,
                       'optimized_2d': False, 'selected_fidelity': '1D', 'two_d_calls': 0,
                       'final_stage': 'none', 'localization': 'none', 'prior_style': 'matern',
                       'anisotropy': args.anisotropy,
                       'rv_ratio_bounds': (sequential.min_ratio, sequential.max_ratio),
                       'rv_hard_cap_ohmm': RV_CAP_OHMM, 'rh_cap_ohmm': RH_CAP_OHMM,
                       'parameter_units': 'natural_log', 'unified_domain': True,
                       'posterior_prediction_kind': 'forward', 'posterior_misfit_kind': 'forward',
                       'posterior_forward_verified': True,
                       'point_elapsed_seconds': record['seconds']}
            with open(f'inversion_results_assim_{index}.pkl', 'xb') as stream:
                pickle.dump(results, stream)
            previous_index = index
            note = ' [gate open, reverted]' if reverted else ''
            if cascade is not None:
                note = f" [cascade: best {cascade['best_member_misfit']:.3g} after the first share -> 1D]"
            print(f"Index {index}: move {tuple(int(m) for m in movement)}, entered {entered} cells, "
                  f"rho {float(np.mean(rho)):.3f}{note}; "
                  f"1D misfit mean-pred {steps[0]['mean_prediction_1d_misfit']:.4g} -> "
                  f"{steps[-1]['mean_prediction_1d_misfit']:.4g}, member median "
                  f"{steps[-1]['member_1d_misfit_median']:.4g}; Rh RMSE "
                  f"{steps[0]['rh_rmse_log10']:.3f} -> {steps[-1]['rh_rmse_log10']:.3f}; "
                  f"2P/n Rh {posterior_diagnostics['matern_2p_over_n_rh_median']:.2f}; "
                  f"{record['seconds']:.0f} s", flush=True)
    finally:
        os.chdir(previous_cwd)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
