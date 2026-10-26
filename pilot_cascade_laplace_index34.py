"""Index-34 cascade ES-MDA with a Laplace final stage.

250 fresh members of the index-31 prior recipe (fast_gaussian 50/50-ft Rh,
independent AR1 latent, caps 40/40). Ten linear MDA steps: step 1 assimilates
with the 0D simulator, steps 2-5 with the 1D simulator (50% of the likelihood
weight). The best-misfit member then starts a MAP optimization that consumes
the remaining 50% (data variance inflated by 2), and 250 Laplace draws are
forward-verified on HPC. Curtains are rendered after the 0D step, after the
1D steps, and for the final posterior.
"""

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esmda_core import (LocalWindow, clamp_log_rh, measurement_misfit,
                        noise_floor_mask, observation_vectors)
from esmda_remote2d import SimBackendConfig, make_remote2d
from background_prior import BackgroundPrior
from rml import run_rml, gaussian_moment_redraw
from distance_localization import (FurrerBengtssonLocalization, blend_localized_samples,
                                   localized_rtps)
from geostat.gaussian_sim import fast_gaussian
from lm_laplace import SpatialPrior, optimize, sample_laplace
from plot_curtain import posterior_moments
from rolling_prior import RollingPriorBuffer
from plot_index31_posterior_curtain import benchmark_curtain
import run_DA_sequential as sequential
from pilot_2d_only_index31 import RH_CORRELATION_X_FT, RH_CORRELATION_Z_FT, cap_rh, \
    rh_prior_moments, write_npz
from transition_mda_study import (DEFAULT_REFERENCE, checkpoint, inflation_schedule,
                                  reference_truth, rh_rmse, update)

ROOT = Path(__file__).resolve().parent
INDEX = 34
MEMBERS = 250
LOCALIZATION_THRESHOLD = .3
MAP_INFLATION = 2.


def latent_prior_moments(reference_manifest, anisotropy, anisotropy_mean=.5, anisotropy_std=.5):
    """(mean, std) of the anisotropy state: probit latent N(0, latent_std) or ln(Rv/Rh)."""
    if anisotropy == 'log_clipped':
        return float(anisotropy_mean), float(anisotropy_std)
    return 0., float(reference_manifest['rv_parameterization']['latent_std'])


def matern_reference(reference_manifest, anisotropy='probit', anisotropy_mean=.5, anisotropy_std=.5):
    """Matern (nu = 1) prior with the cascade's moments; also used to score any ensemble."""
    from matern_prior import MaternPrior
    nx, _, nz = reference_manifest['dims']
    mean_rh, std_rh = rh_prior_moments(reference_manifest)
    latent_mean, latent_std = latent_prior_moments(reference_manifest, anisotropy,
                                                   anisotropy_mean, anisotropy_std)
    return MaternPrior((nx, nz), (reference_manifest['dx_ft'], reference_manifest['dz_ft']),
                       [[RH_CORRELATION_X_FT, RH_CORRELATION_Z_FT],
                        [reference_manifest['prior_latent_correlation_x_ft'],
                         reference_manifest['prior_latent_correlation_z_ft']]],
                       [mean_rh, latent_mean], [std_rh, latent_std])


def ensemble_diagnostics(state, nx, nz, scorer, anisotropy):
    """Window/bit-column ensemble-mean ln Rh, mean physical ln(Rv/Rh), Matern 2P/n per block."""
    from esmda_core import log_ratio_from_latent
    cells = nx * nz
    mean_rh = state[:cells].mean(axis=1).reshape(nx, nz)
    ratio = log_ratio_from_latent(state[cells:], 1., float(sequential.max_ratio), anisotropy)
    penalty = [2 * scorer.penalty(state, (block,)) / cells for block in (0, 1)]
    return {'window_mean_ln_rh': float(mean_rh.mean()),
            'bit_column_mean_ln_rh': float(mean_rh[nx // 2].mean()),
            'mean_ln_rv_over_rh': float(ratio.mean()),
            'isotropic_fraction': float(np.mean(ratio <= 0.)),
            'matern_2p_over_n_rh_median': float(np.median(penalty[0])),
            'matern_2p_over_n_latent_median': float(np.median(penalty[1]))}


def member_misfit_summary(predictions, observations, covariance):
    """1D misfit of the mean prediction and per-member quantiles (comparable across fidelities)."""
    misfits = np.mean((predictions - observations[:, None]) ** 2 / covariance[:, None], axis=0)
    p10, p50, p90 = np.percentile(misfits, [10, 50, 90])
    return {'mean_prediction_1d_misfit': measurement_misfit(predictions.mean(axis=1), observations,
                                                            covariance),
            'member_1d_misfit_p10': float(p10), 'member_1d_misfit_median': float(p50),
            'member_1d_misfit_p90': float(p90)}


def draw_matern_prior(reference_manifest, seed, members, anisotropy='probit',
                      anisotropy_mean=.5, anisotropy_std=.5):
    """Matern (nu = 1) SPDE draws; the current lengths are used as practical ranges."""
    prior = matern_reference(reference_manifest, anisotropy, anisotropy_mean, anisotropy_std)
    state = clamp_log_rh(prior.sample(members, seed))
    return cap_rh(state, 40.), prior


def draw_prior(reference_manifest, seed, members, anisotropy='probit',
               anisotropy_mean=.5, anisotropy_std=.5):
    nx, _, nz = reference_manifest['dims']
    mean_rh, std_rh = rh_prior_moments(reference_manifest)
    latent_mean, latent_std = latent_prior_moments(reference_manifest, anisotropy,
                                                   anisotropy_mean, anisotropy_std)
    np.random.seed(seed)
    rh = mean_rh + fast_gaussian(
        np.array([nz, nx]), np.array([std_rh]),
        np.array([RH_CORRELATION_Z_FT / reference_manifest['dz_ft'],
                  RH_CORRELATION_X_FT / reference_manifest['dx_ft']]), num_samples=members)
    np.random.seed(seed + 1)
    latent = latent_mean + latent_std * sequential._ar1_field(
        nx, nz, members,
        np.exp(-reference_manifest['dx_ft'] / reference_manifest['prior_latent_correlation_x_ft']),
        np.exp(-reference_manifest['dz_ft'] / reference_manifest['prior_latent_correlation_z_ft']))
    state = clamp_log_rh(np.vstack([rh, latent.reshape(nx * nz, members)]))
    return cap_rh(state, 40.)


def cascade_inflation(steps=10, tempering='linear-inflation', likelihood_weight=.5):
    """Normalize linear inflation or equal-energy steps to the requested likelihood weight."""
    if not 1 <= steps <= 10:
        raise ValueError('Require 1..10 MDA steps')
    if not np.isfinite(likelihood_weight) or not 0 < likelihood_weight <= 1:
        raise ValueError('Likelihood weight must lie in (0,1]')
    if tempering == 'equal-energy':
        alpha = np.full(steps, steps / likelihood_weight)
    elif tempering == 'linear-inflation':
        taper = np.arange(15., 15. - steps, -1.)
        alpha = taper * (np.reciprocal(taper).sum() / likelihood_weight)
    else:
        raise ValueError(f'Unknown MDA tempering {tempering}')
    assert alpha.shape == (steps,) and np.isclose(np.reciprocal(alpha).sum(), likelihood_weight)
    return alpha


def original_background_prior(manifest):
    nx, _, nz = manifest['dims']
    mean_rh, std_rh = rh_prior_moments(manifest)
    return BackgroundPrior(
        (nx, nz), (manifest['dx_ft'], manifest['dz_ft']),
        ((RH_CORRELATION_X_FT, RH_CORRELATION_Z_FT),
         (manifest['prior_latent_correlation_x_ft'], manifest['prior_latent_correlation_z_ft'])),
        (mean_rh, 0.), (std_rh, manifest['rv_parameterization']['latent_std']))


def select_best_member(predictions, observations, covariance):
    misfits = np.array([measurement_misfit(predictions[:, member], observations, covariance)
                        for member in range(predictions.shape[1])])
    return int(np.argmin(misfits)), misfits


def restart_ensemble(state, movement, reference_manifest):
    """Reindex a saved posterior window, conditioning entered cells on the prior."""
    ln10 = np.log(10.)
    shifted = sequential.shift_window_state(
        state, movement,
        reference_manifest['prior_background'] * ln10,
        reference_manifest['prior_reservoir'] * ln10,
        reference_manifest['prior_mode_std'] * ln10,
        reference_manifest['rv_parameterization']['latent_std'],
        reference_manifest['prior_switch_prob'], prior_style='gaussian',
        fraction=reference_manifest['prior_reservoir_fraction'],
        dx=reference_manifest['dx_ft'], dz=reference_manifest['dz_ft'],
        corr_x_ft=RH_CORRELATION_X_FT, corr_z_ft=RH_CORRELATION_Z_FT,
        latent_corr_z_ft=reference_manifest['prior_latent_correlation_z_ft'],
        latent_corr_x_ft=reference_manifest['prior_latent_correlation_x_ft'])
    return cap_rh(shifted, 40.)


def render_curtain(output, fields, study_meta, reference, truth_pkg, title):
    benchmark = truth_pkg['benchmark']
    truth, x_edges, z_edges, well, _ = benchmark_curtain(benchmark, study_meta)
    manifest_bounds = truth_pkg['ratio_bounds']
    cap = truth_pkg['rv_cap']
    mean_rh, mean_rv, variance_rh, variance_rv, _ = posterior_moments(
        fields.T, study_meta['dims'], manifest_bounds, cap, truth_pkg.get('anisotropy', 'probit'))
    moments = {'Rh': (mean_rh / np.log(10.), np.sqrt(variance_rh)),
               'Rv': (mean_rv / np.log(10.), np.sqrt(variance_rv))}
    ref_logs = {name: np.log10(truth[name]) for name in ('Rh', 'Rv')}
    values = np.concatenate([ref_logs['Rh'].ravel(), ref_logs['Rv'].ravel(),
                             moments['Rh'][0].ravel(), moments['Rv'][0].ravel()])
    vmin, vmax = np.percentile(values, [1, 99])
    std_max = max(float(np.percentile(moments[name][1], 99)) for name in ('Rh', 'Rv'))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True,
                             sharex=True, sharey=True)
    image = spread = None
    for row, name in enumerate(('Rh', 'Rv')):
        model, std = moments[name]
        for col, (array, label) in enumerate(((ref_logs[name], 'Benchmark'),
                                              (model, 'Posterior mean'),
                                              (std, 'Posterior std'))):
            ax = axes[row, col]
            artist = ax.pcolormesh(x_edges, z_edges, array.T, shading='flat',
                                   cmap='magma' if col == 2 else 'viridis',
                                   vmin=0 if col == 2 else vmin,
                                   vmax=std_max if col == 2 else vmax)
            if col == 2:
                spread = artist
            else:
                image = artist
            ax.plot(well[0], well[1], 'x', color='white', ms=7, mew=1.7)
            ax.set_title(f'{label} — {name}')
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(z_edges[-1], z_edges[0])
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X [ft]')
            if col == 0:
                ax.set_ylabel('TVD [ft]')
    fig.colorbar(image, ax=axes[:, :2].ravel().tolist(), shrink=.72,
                 label='log10 resistivity [ohm m]')
    fig.colorbar(spread, ax=axes[:, 2].ravel().tolist(), shrink=.72,
                 label='Posterior std of log10 resistivity')
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    return str(output)


def main(argv=None):
    defaults = SimBackendConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    defaults.add_cli(parser)
    parser.set_defaults(concurrency=25)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--strategy', choices=['cascade', 'independent-reset', 'rml-reset'], default='cascade')
    parser.add_argument('--rml-members', type=int, default=20)
    parser.add_argument('--index', type=int, default=INDEX)
    parser.add_argument('--ensemble-size', type=int, default=MEMBERS)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument('--max-iterations', type=int, default=25)
    parser.add_argument('--max-trials', type=int, default=10)
    parser.add_argument('--max-seconds', type=float, default=28800.)
    parser.add_argument('--max-step', type=float, default=.5)
    parser.add_argument('--relative-data-improvement', type=float, default=.1)
    parser.add_argument('--zero-d-steps', type=int, default=1,
                        help='Leading MDA steps assimilated with the 0D simulator')
    parser.add_argument('--mda-steps', type=int, default=10,
                        help='Number of MDA steps (1..10); jointly spend 50%% of the likelihood')
    parser.add_argument('--prior-family', choices=['fast_gaussian', 'matern'], default='fast_gaussian',
                        help='Cascade prior draws: fast_gaussian recipe or Matern nu=1 SPDE (sparse precision)')
    parser.add_argument('--anisotropy', choices=['probit', 'log_clipped'], default='probit',
                        help='Anisotropy state: probit latent or ln(Rv/Rh) clipped at 0')
    parser.add_argument('--anisotropy-mean', type=float, default=.5,
                        help='log_clipped: prior mean of ln(Rv/Rh)')
    parser.add_argument('--anisotropy-std', type=float, default=.5,
                        help='log_clipped: prior std of ln(Rv/Rh)')
    parser.add_argument('--mda-likelihood-weight', type=float, default=.5,
                        help='Likelihood weight spent by the cascade MDA steps (cascade strategy); '
                             '1.0 assimilates all data energy in 0D/1D')
    parser.add_argument('--tempering', choices=['linear-inflation', 'equal-energy'],
                        default='linear-inflation',
                        help='Equal-energy spends the same likelihood weight at every MDA step')
    parser.add_argument('--mda-only', action='store_true',
                        help='Plot the MDA posterior curtain and exit before the 2D stage')
    parser.add_argument('--restart-state', type=Path, default=None,
                        help='Saved ensemble .npz (fields, member_ids) to restart from')
    parser.add_argument('--prior-buffer-ft', type=float, default=0.,
                        help='Data-free prior halo on each side; 0 uses the legacy extension')
    parser.add_argument('--localization', choices=['adaptive', 'fb'], default='adaptive')
    parser.add_argument('--localization-range-x-ft', type=float, default=60.)
    parser.add_argument('--localization-range-z-ft', type=float, default=150.)
    parser.add_argument('--rtps-strength', type=float, default=0.,
                        help='Localized spread relaxation after the Laplace blend, in [0,1]')
    parser.add_argument('--restart-from-index', type=int, default=None,
                        help='Logging index the restart ensemble is expressed in')
    parser.add_argument('--gate-threshold', type=float, default=None,
                        help='Engage the 2D MAP/Laplace stage only if the ensemble-mean '
                             '1D reduced misfit after the initial MDA steps exceeds this')
    parser.add_argument('--gate-extra-steps', type=int, default=5,
                        help='Additional 1D MDA steps (same tempering construction, spending '
                             'the remaining 50%%) when the gate stays closed')
    parser.add_argument('--gated-map-iterations', type=int, default=10,
                        help='Optimizer iteration cap when the gate engages the 2D stage')
    args = parser.parse_args(argv)
    independent = args.strategy in ('independent-reset', 'rml-reset')
    if independent:
        if (args.mda_steps != 10 or args.zero_d_steps != 1 or args.ensemble_size != 250
                or args.rtps_strength != 0 or args.mda_only
                or args.tempering != 'linear-inflation' or args.gate_threshold is None):
            parser.error('Independent reset requires 1x0D + 9x1D, 250 members, and RTPS disabled')
        args.gate_extra_steps = 0
    if args.strategy == 'rml-reset' and args.rml_members != 20:
        parser.error('This RML strategy requires 20 optimizers')
    if not np.isfinite(args.rtps_strength) or not 0 <= args.rtps_strength <= 1:
        parser.error('--rtps-strength must lie in [0,1]')
    if args.rtps_strength > 0 and args.localization != 'fb':
        parser.error('Localized RTPS requires --localization fb')
    if not np.isfinite(args.prior_buffer_ft) or args.prior_buffer_ft < 0:
        parser.error('--prior-buffer-ft must be nonnegative')
    if (args.index < 0 or args.ensemble_size < 2 or args.seed < 0
            or not 1 <= args.max_iterations <= 25 or args.max_trials < 1
            or args.max_seconds <= 0 or args.max_step <= 0
            or not 0 < args.relative_data_improvement < 1
            or not 0 <= args.zero_d_steps <= args.mda_steps <= 10
            or not 0 <= args.gate_extra_steps <= 10
            or not 1 <= args.gated_map_iterations <= 25
            or (args.gate_threshold is not None
                and (not np.isfinite(args.gate_threshold) or args.gate_threshold <= 0))):
        parser.error('Invalid cascade configuration')
    if (args.restart_state is None) != (args.restart_from_index is None):
        parser.error('Use both --restart-state and --restart-from-index, or neither')
    if args.restart_from_index is not None and not 0 <= args.restart_from_index < args.index:
        parser.error('Restart index must precede the inversion index')
    if args.sim_backend != 'slurm':
        parser.error('The 2D Laplace stage requires --sim-backend slurm')

    reference_dir = args.reference.resolve()
    reference = checkpoint(reference_dir, args.index)
    manifest = json.loads((reference_dir / 'run.json').read_text())
    benchmark = Path(manifest['benchmark'])
    truth = reference_truth(benchmark)
    nx, ny, nz = manifest['dims']
    dx, dy, dz = manifest['spacing_ft']
    cells = nx * nz
    sequential.dims = (nx, ny, nz)
    sequential.dX, sequential.dY, sequential.dZ = dx, dy, dz
    sequential.max_ratio = float(manifest['prior_ratio_max'])
    sequential.log_rv_cap = float(np.log(40.))
    sequential.anisotropy = args.anisotropy
    sequential.wellpath = truth[3]
    window = LocalWindow((nx, ny, nz), (dx, dy, dz))
    shift = None
    source_shift = None
    for point in range(args.index + 1):
        position = tuple(truth[3][key][point, 0] * sequential.meter_to_feet
                         for key in ('X', 'Y', 'Z'))
        shift, _ = window.update(position)
        if args.restart_from_index == point:
            source_shift = dict(shift)
    if not all(np.isclose(shift[key], reference['shift_ft'][key], rtol=0, atol=1e-6)
               for key in ('x', 'y', 'z')):
        raise ValueError(f'Index-{args.index} moving-window geometry disagrees with reference')
    sequential.current_shift = shift
    data = pd.read_pickle(ROOT / manifest['data'])
    variances = pd.read_pickle(ROOT / manifest['variance'])
    tools = list(data.columns)
    observations, covariance = observation_vectors(data, variances, args.index, tools)
    shallow = np.zeros(len(observations), dtype=bool)
    shallow[sequential.component_positions(len(observations))] = True
    mask = ~(noise_floor_mask(observations, covariance) | shallow)
    if not np.array_equal(mask, reference['observation_mask']) or not np.array_equal(
            observations[mask], reference['observations']):
        raise ValueError(f'Index-{args.index} observations or channel mask changed')
    observations, covariance = observations[mask], covariance[mask]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    distance_loc = None
    if args.localization == 'fb':
        bit_z = int(np.floor((truth[3]['Z'][args.index, 0] * sequential.meter_to_feet
                              - shift['z']) / dz))
        distance_loc = FurrerBengtssonLocalization(
            (nx, nz), (dx, dz), (nx // 2, bit_z),
            (args.localization_range_x_ft, args.localization_range_z_ft),
            args.ensemble_size, len(observations), args.index, output / 'localization.pkl')
        write_npz(output / 'localization-mask.npz', weights=distance_loc.weights,
                  protected=distance_loc.weights == 0)
    config = defaults.from_args(args)
    if not 0 < args.mda_likelihood_weight <= 1:
        parser.error('--mda-likelihood-weight must lie in (0, 1]')
    inflation = cascade_inflation(args.mda_steps, args.tempering,
                                   likelihood_weight=1. if independent else args.mda_likelihood_weight)
    map_inflation = 1. if independent else MAP_INFLATION
    restart = None
    if args.restart_state is not None:
        restart_path = args.restart_state.resolve()
        with np.load(restart_path) as saved_state:
            if 'fields' not in saved_state.files:
                raise ValueError('Restart state must contain a fields array')
            restart_fields = np.asarray(saved_state['fields'], dtype=float)
        if (restart_fields.shape != (2 * cells, args.ensemble_size)
                or not np.isfinite(restart_fields).all()):
            raise ValueError('Restart ensemble shape is invalid for this configuration')
        movement = (int(round((shift['x'] - source_shift['x']) / dx)),
                    int(round((shift['z'] - source_shift['z']) / dz)))
        restart = {'source_state': str(restart_path),
                   'source_sha256': hashlib.sha256(restart_path.read_bytes()).hexdigest(),
                   'restart_from_index': args.restart_from_index,
                   'movement_cells': list(movement),
                   'entered_cell_fill': 'gaussian conditional draws; Rh 50/50-ft, '
                                        'latent 80/35-ft correlation; Rh cap 40 applied'}
    study = {'method': ('cascade_mda_laplace_restart' if restart else 'cascade_mda_laplace'),
             'logging_index': args.index,
             'reference': str(reference_dir), 'ensemble_size': args.ensemble_size,
             'seed': args.seed,
             'prior': 'fast_gaussian 50/50-ft Rh + independent AR1 latent; '
                      'clamp [-1,4] then Rh cap 40',
             'prior_ln_mean': float(rh_prior_moments(manifest)[0]),
             'prior_ln_variance': float(rh_prior_moments(manifest)[1] ** 2),
             'rh_cap_ohmm_effective': 40., 'rv_cap_ohmm_effective': 40.,
             'schedule': (f'{args.zero_d_steps} x 0D then '
                          f'{args.mda_steps - args.zero_d_steps} x 1D at 50% total '
                          'MDA weight, MAP+Laplace final 50% (2xCd)'),
             'mda_steps_executed': args.mda_steps,
             'zero_d_steps': args.zero_d_steps,
             'tempering': args.tempering,
             'covariance_inflation': inflation.tolist(),
             'mda_likelihood_weight': float(np.reciprocal(inflation).sum()),
             'laplace_likelihood_weight': 1. / MAP_INFLATION,
             'rh_cap_after_each_mda_update': True,
             'localization': {'type': 'sigm', 'correlation_threshold': LOCALIZATION_THRESHOLD},
             'map': {'max_iterations': args.max_iterations, 'max_trials': args.max_trials,
                     'max_step': args.max_step,
                     'relative_data_improvement': args.relative_data_improvement,
                     'max_seconds': args.max_seconds, 'inflation': MAP_INFLATION,
                     'prior': 'SpatialPrior(ensemble), Rh lengths 50/50, latent 80/35'},
              'dims': [nx, ny, nz], 'spacing_ft': [dx, dy, dz],
              'shift_ft': {name: float(shift[name]) for name in ('x', 'y', 'z')},
              'simulator': asdict(config)}
    if distance_loc is not None:
        study['localization'] = {
            'type': 'distance', 'taper': 'fb', 'implementation': 'PET Furrer-Bengtsson',
            'range_x_ft': args.localization_range_x_ft,
            'range_z_ft': args.localization_range_z_ft,
            'bit_cell': [nx // 2, bit_z], 'field_order': '[parameter_block=2, x, z]; C-order',
            'protected_parameters': int(np.sum(distance_loc.weights == 0)),
            'stages': ['0D', '1D', 'MAP step', 'member-wise Laplace blend']}
        if independent:
            study['localization']['stages'] = ['0D', '1D']
    study['strategy'] = args.strategy
    study['prior_family'] = args.prior_family
    study['anisotropy'] = {'parameterization': args.anisotropy,
                           'state_moments': list(latent_prior_moments(
                               manifest, args.anisotropy, args.anisotropy_mean, args.anisotropy_std)),
                           'map': ('ln(Rv/Rh) = max(state, 0); Rv cap 40' if args.anisotropy == 'log_clipped'
                                   else 'ln(Rv/Rh) = ln(max_ratio) Phi(state); Rv cap 40')}
    if args.prior_family == 'matern':
        study['prior'] = ('Matern nu=1 SPDE (sparse precision, Robin, cellwise std); practical ranges '
                          f'Rh {RH_CORRELATION_X_FT:g}/{RH_CORRELATION_Z_FT:g} ft, anisotropy '
                          f"{manifest['prior_latent_correlation_x_ft']:g}/"
                          f"{manifest['prior_latent_correlation_z_ft']:g} ft; clamp [-1,4] then Rh cap 40")
    study['zero_d_steps'] = args.zero_d_steps
    study['schedule'] = (f'{args.zero_d_steps} x 0D then {args.mda_steps - args.zero_d_steps} x 1D, '
                         f'MDA likelihood weight {float(np.reciprocal(inflation).sum()):g}')
    if independent:
        study['method'] = 'full_energy_mda_independent_background_2d'
        study['schedule'] = '1x0D + 9x1D, full likelihood; independent 2D reset if misfit > gate'
        study['laplace_likelihood_weight'] = 1.
        study['map']['inflation'] = 1.
        study['map']['prior'] = 'original stationary Gaussian: Rh squared-exponential 50/50 ft; latent exponential 80/35 ft'
        study['map']['max_iterations'] = args.gated_map_iterations
        study['posterior_transfer'] = 'direct bounded Laplace samples; independent of failed MDA except starting model'
        if args.strategy == 'rml-reset':
            study['method'] = 'full_energy_mda_independent_rml_gaussian_redraw'
            study['posterior_transfer'] = '250 Gaussian draws from cellwise mean/variance of 20 RML fits'
            study['rml'] = {'members': args.rml_members, 'starting_models': 'top 20 final 1D members',
                            'randomization': 'independent prior centers and N(0,Cd) measurement noise',
                            'data_inflation': 1., 'parallelism': 'batched Slurm forward/adjoint waves',
                            'max_iterations_per_optimizer': args.gated_map_iterations}
            study.pop('laplace_likelihood_weight')
            study['rml_likelihood_weight'] = 1.
    study['rtps'] = {'strength': args.rtps_strength, 'stage': 'after Laplace blend',
                     'reference': 'SpatialPrior.std from ten-step MDA ensemble, including floors',
                     'localization': 'same PET fb weights',
                     'bounds': 'reapplied after inflation; achieved spread and mean shift recorded'}
    if independent:
        study['rtps'] = {'strength': 0., 'stage': 'disabled for independent posterior'}
    if restart is not None:
        study['restart'] = restart
    if args.prior_buffer_ft > 0:
        study['prior_buffer'] = {
            'width_ft': args.prior_buffer_ft,
            'rh_kernel': 'squared-exponential', 'latent_kernel': 'exponential',
            'conditioning': 'all retained data-free buffer columns/rows; separable covariance',
            'updated_by_data': False, 'checkpoint': 'prior-buffer.npz'}
        study['prior'] = 'rolling data-free Gaussian prior buffer; Rh 50/50-ft, AR1 latent 80/35-ft'
        if restart is not None:
            restart['entered_cell_fill'] = 'persistent data-free prior buffer; posterior overlap retained'
    if args.gate_threshold is not None:
        study['gate'] = {'threshold': args.gate_threshold,
                         'extra_1d_steps': args.gate_extra_steps,
                         'engaged_map_iterations': args.gated_map_iterations,
                         'skip_posterior_forecast_when_engaged': True,
                         'criterion': 'ensemble-mean 1D reduced misfit (original Cd) '
                                       'after the initial MDA steps'}
        if args.strategy == 'rml-reset':
            study['gate']['skip_posterior_forecast_when_engaged'] = False
    (output / 'study.json').write_text(json.dumps(study, indent=2))
    truth_pkg = {'benchmark': benchmark, 'ratio_bounds': manifest['rv_parameterization']['ratio_bounds'],
                 'rv_cap': 40., 'anisotropy': args.anisotropy}
    study_meta = {'dims': [nx, ny, nz], 'spacing_ft': [dx, dy, dz], 'shift_ft': study['shift_ft'],
                  'logging_index': args.index}

    if args.prior_buffer_ft > 0:
        lengths = ((RH_CORRELATION_X_FT, RH_CORRELATION_Z_FT),
                   (manifest['prior_latent_correlation_x_ft'],
                    manifest['prior_latent_correlation_z_ft']))
        means = (rh_prior_moments(manifest)[0], 0.)
        stds = (rh_prior_moments(manifest)[1], manifest['rv_parameterization']['latent_std'])
        if restart is None:
            buffer = RollingPriorBuffer.create(
                (nx, nz), (dx, dz), lengths, means, stds, args.ensemble_size,
                args.prior_buffer_ft, args.seed)
            state = buffer.active_fields().reshape(2 * cells, args.ensemble_size)
        else:
            buffer = RollingPriorBuffer.load(restart_path.parent / 'prior-buffer.npz')
            expected_padding = tuple(int(np.ceil(args.prior_buffer_ft / step)) for step in (dx, dz))
            if (buffer.active_shape != (nx, nz) or buffer.spacing != (dx, dz)
                    or buffer.padding != expected_padding or buffer.lengths != lengths
                    or buffer.means != means or buffer.stds != stds
                    or buffer.fields.shape[-1] != args.ensemble_size):
                raise ValueError('Restart prior buffer does not match the run configuration')
            state = buffer.restart(restart_fields, tuple(restart['movement_cells']))
        state = cap_rh(clamp_log_rh(state), 40.)
        buffer.save(output / 'prior-buffer.npz')
    elif restart is None and args.prior_family == 'matern':
        state, matern = draw_matern_prior(manifest, args.seed, args.ensemble_size, args.anisotropy,
                                          args.anisotropy_mean, args.anisotropy_std)
    elif restart is None:
        state = draw_prior(manifest, args.seed, args.ensemble_size, args.anisotropy,
                           args.anisotropy_mean, args.anisotropy_std)
    else:
        state = restart_ensemble(restart_fields, tuple(restart['movement_cells']), manifest)
    if state.shape != (2 * cells, args.ensemble_size) or not np.isfinite(state).all():
        raise ValueError('Invalid starting ensemble')
    protected_reference = None if distance_loc is None else state[distance_loc.weights == 0].copy()
    scorer = matern_reference(manifest, args.anisotropy, args.anisotropy_mean, args.anisotropy_std)
    if restart is None and not independent:
        write_npz(output / 'state-00-prior.npz', fields=state, member_ids=np.arange(args.ensemble_size))

    def collar_change(fields):
        if distance_loc is None:
            return None
        change = float(np.max(np.abs(fields[distance_loc.weights == 0] - protected_reference)))
        if change != 0.:
            raise ValueError('An assimilation stage changed the protected collar')
        return change

    if restart is not None:
        write_npz(output / 'state-00-restart.npz', fields=state,
                  member_ids=np.arange(args.ensemble_size))
    elif independent:
        write_npz(output / 'state-00-prior.npz', fields=state,
                  member_ids=np.arange(args.ensemble_size))
    rng = np.random.default_rng(args.seed + 5000)
    history = []
    gate = None
    old_cwd = Path.cwd()

    try:
        os.chdir(output)
        sequential.tools = [str(tool) for tool in tools]
        sequential.setup_simulators(benchmark, sequential.tools)
        adapter = make_remote2d(config, (nx, ny, nz), dx, dy, dz,
                                [ast.literal_eval(str(tool)) for tool in tools])
        adapter.jobs_dir = str(output / 'jobs')
        if not adapter.backend().trusts_ad('2D'):
            raise RuntimeError('A trusted exact-adjoint 2D build is required')

        def predict_0d():
            initial = np.vstack([state[:cells].mean(axis=0), state[cells:].mean(axis=0)])
            return initial, np.column_stack([
                sequential.simulate_pred_only(initial[:, j], args.index, '0D')[mask]
                for j in range(initial.shape[1])])

        def predict_1d():
            columns = state.reshape(2, nx, nz, -1)[:, nx // 2].reshape(2 * nz, -1)
            return np.column_stack([
                sequential.simulate_pred_only(columns[:, j], args.index, '1D')[mask]
                for j in range(columns.shape[1])])

        if restart is None:
            prior_prediction = predict_1d()
            history.append({'stage': 'prior', 'mda_step': 0, 'remaining_weight': float(np.reciprocal(inflation).sum()),
                            'ensemble_mean_reduced_misfit': measurement_misfit(
                                prior_prediction.mean(axis=1), observations, covariance),
                            'misfit_fidelity': '1D', 'rh_rmse_log10': rh_rmse(state, reference, truth),
                            **member_misfit_summary(prior_prediction, observations, covariance),
                            **ensemble_diagnostics(state, nx, nz, scorer, args.anisotropy)})
            print(f"Prior: 1D misfit {history[-1]['ensemble_mean_reduced_misfit']:.4g}, "
                  f"RMSE {history[-1]['rh_rmse_log10']:.4g}", flush=True)
        if restart is not None:
            restart['initial_ensemble_1d_misfit'] = measurement_misfit(
                predict_1d().mean(axis=1), observations, covariance)
            print(f"Restart ensemble 1D misfit at index {args.index}: "
                  f"{restart['initial_ensemble_1d_misfit']:.4g}", flush=True)

        for step in range(args.mda_steps):
            fidelity = '0D' if step < args.zero_d_steps else '1D'
            started = time.monotonic()
            if fidelity == '0D':
                _, predicted = predict_0d()
            else:
                predicted = predict_1d()
            state = cap_rh(update(state, predicted, observations, covariance,
                                  inflation[step], rng, cells,
                                  None if fidelity == '0D' else LOCALIZATION_THRESHOLD,
                                  distance_localization=distance_loc), 40.)
            protected_change = collar_change(state)
            post_prediction = predict_0d()[1] if fidelity == '0D' else predict_1d()
            stage = {'stage': fidelity, 'mda_step': step + 1,
                     'alpha': float(inflation[step]),
                     'remaining_weight': float(np.reciprocal(inflation[step + 1:]).sum()),
                     'seconds': time.monotonic() - started,
                     'ensemble_mean_reduced_misfit': measurement_misfit(
                         post_prediction.mean(axis=1), observations, covariance),
                      'rh_rmse_log10': rh_rmse(state, reference, truth)}
            if distance_loc is not None:
                stage['protected_collar_max_change'] = protected_change
            stage.update(member_misfit_summary(predict_1d() if fidelity == '0D' else post_prediction,
                                               observations, covariance))
            stage.update(ensemble_diagnostics(state, nx, nz, scorer, args.anisotropy))
            if step == 0:
                write_npz(output / f'state-01-{fidelity.lower()}.npz', fields=state,
                          member_ids=np.arange(args.ensemble_size))
                stage['curtain'] = render_curtain(
                    output / f'curtain-01-after-{fidelity.lower()}.png', state, study_meta, reference,
                    truth_pkg, f"Index {args.index} • after {fidelity} MDA step 1/{args.mda_steps} • "
                               f"misfit {stage['ensemble_mean_reduced_misfit']:.4g}")
            history.append(stage)
            print(f"Step {step + 1} ({fidelity}): remaining weight "
                  f"{stage['remaining_weight']:.3f}, RMSE {stage['rh_rmse_log10']:.4g}",
                  flush=True)
        last_fidelity = '0D' if args.zero_d_steps >= args.mda_steps else '1D'
        predicted_final = predict_0d()[1] if last_fidelity == '0D' else predict_1d()
        history[-1]['ensemble_mean_reduced_misfit'] = measurement_misfit(
            predicted_final.mean(axis=1), observations, covariance)
        history[-1]['curtain'] = render_curtain(
            output / f'curtain-{args.mda_steps:02d}-after-{last_fidelity.lower()}.png',
            state, study_meta, reference,
            truth_pkg, f"Index {args.index} • after {last_fidelity} MDA steps "
                       f"{args.mda_steps}/10 • "
                       f"misfit {history[-1]['ensemble_mean_reduced_misfit']:.4g}")
        write_npz(output / f'state-{args.mda_steps:02d}-{last_fidelity.lower()}.npz', fields=state,
                  member_ids=np.arange(args.ensemble_size))
        print(f"{last_fidelity} stage complete: "
              f"misfit {history[-1]['ensemble_mean_reduced_misfit']:.4g}", flush=True)
        if args.gate_threshold is not None:
            gate = {'threshold': args.gate_threshold,
                    'post_mda_reduced_misfit': history[-1]['ensemble_mean_reduced_misfit'],
                    'extra_1d_steps': args.gate_extra_steps}
            gate['engaged'] = bool(gate['post_mda_reduced_misfit'] > gate['threshold'])
            print(f"Gate: post-MDA misfit {gate['post_mda_reduced_misfit']:.4g} vs "
                  f"{gate['threshold']:g} -> "
                  f"{'2D MAP/Laplace engaged' if gate['engaged'] else '1D completion path'}",
                  flush=True)
        if args.mda_only:
            # Final-stage predictions per member (original Cd), for ranking hand-offs.
            write_npz(output / 'mda-final-predictions.npz', predictions=predicted_final,
                      observations=observations, covariance=covariance, mask=mask,
                      fidelity=last_fidelity)
            result = {'status': 'mda_only', 'logging_index': args.index,
                      'mda_steps_executed': args.mda_steps,
                      'final_stage': last_fidelity,
                      'final_reduced_misfit': history[-1]['ensemble_mean_reduced_misfit'],
                      'final_rh_rmse_log10': history[-1]['rh_rmse_log10'],
                      'remaining_weight': history[-1]['remaining_weight'],
                      'mda_history': history}
            if gate is not None:
                result['gate'] = gate
            if restart is not None:
                result['restart'] = restart
            (output / 'result.json').write_text(json.dumps(result, indent=2))
            print('MDA-only run complete; skipping the 2D stage', flush=True)
            return
        if gate is not None and not gate['engaged']:
            extra_inflation = (cascade_inflation(args.gate_extra_steps, args.tempering)
                               if args.gate_extra_steps else [])
            for extra_step, alpha in enumerate(extra_inflation):
                started = time.monotonic()
                predicted = predict_1d()
                state = cap_rh(update(state, predicted, observations, covariance,
                                      float(alpha), rng, cells, LOCALIZATION_THRESHOLD,
                                      distance_localization=distance_loc), 40.)
                protected_change = collar_change(state)
                post_prediction = predict_1d()
                history.append({'stage': '1D',
                                'mda_step': args.mda_steps + extra_step + 1,
                                'alpha': float(alpha),
                                'remaining_weight': float(np.reciprocal(
                                    extra_inflation[extra_step + 1:]).sum()),
                                'seconds': time.monotonic() - started,
                                'ensemble_mean_reduced_misfit': measurement_misfit(
                                    post_prediction.mean(axis=1), observations, covariance),
                                'rh_rmse_log10': rh_rmse(state, reference, truth)})
                if distance_loc is not None:
                    history[-1]['protected_collar_max_change'] = protected_change
                print(f"Step {args.mda_steps + extra_step + 1} (1D extra): "
                      f"misfit {history[-1]['ensemble_mean_reduced_misfit']:.4g}, "
                      f"RMSE {history[-1]['rh_rmse_log10']:.4g}", flush=True)
            total_steps = args.mda_steps + args.gate_extra_steps
            history[-1]['curtain'] = render_curtain(
                output / f'curtain-{total_steps:02d}-after-1d.png', state, study_meta,
                reference, truth_pkg,
                f"Index {args.index} • 1D completion {total_steps} steps "
                f"(gate {gate['post_mda_reduced_misfit']:.4g} <= {gate['threshold']:g}) • "
                f"misfit {history[-1]['ensemble_mean_reduced_misfit']:.4g}")
            if args.gate_extra_steps:
                write_npz(output / f'state-{total_steps:02d}-1d.npz', fields=state,
                          member_ids=np.arange(args.ensemble_size))
            write_npz(output / 'final_ensemble.npz', fields=state,
                      member_ids=np.arange(args.ensemble_size), kind='1d_mda')
            result = {'status': '1d_completed', 'logging_index': args.index,
                      'strategy': args.strategy,
                      'mda_likelihood_weight': float(np.reciprocal(inflation).sum()),
                      'mda_steps_executed': total_steps,
                      'final_reduced_misfit': history[-1]['ensemble_mean_reduced_misfit'],
                      'final_rh_rmse_log10': history[-1]['rh_rmse_log10'],
                      'gate': gate, 'mda_history': history}
            if restart is not None:
                result['restart'] = restart
            (output / 'result.json').write_text(json.dumps(result, indent=2))
            print(f"1D completion path: {total_steps} steps, "
                  f"misfit {result['final_reduced_misfit']:.4g}", flush=True)
            return
    finally:
        os.chdir(old_cwd)

    best_member, member_misfits = select_best_member(predicted_final, observations, covariance)
    initial = state[:, best_member].copy()
    prior_sp = (original_background_prior(manifest) if independent else
                SpatialPrior(state, (nx, nz), (dx, dz),
                             ((RH_CORRELATION_X_FT, RH_CORRELATION_Z_FT),
                              (manifest['prior_latent_correlation_x_ft'],
                               manifest['prior_latent_correlation_z_ft']))))
    variance_map = covariance * map_inflation
    if independent:
        write_npz(output / 'background-prior.npz', mean=prior_sp.mean, std=prior_sp.std,
                  starting_model=initial)
    lower, upper = -1., float(np.log(40.))

    def physical_model(model):
        physical = np.asarray(model, dtype=float).copy()
        physical[:cells] = np.clip(physical[:cells], lower, upper)
        return physical

    def forward(model):
        response = adapter.run_pred(
            sequential._build_param_dict(physical_model(model))[0],
            args.index, shift, truth[3])[mask]
        if not np.isfinite(response).all():
            raise ValueError('Non-finite 2D forward prediction')
        return response

    def derivative(model):
        parameters, rh, latent = sequential._build_param_dict(physical_model(model))
        prediction, j_rh, j_rv = adapter.run_grad(parameters, args.index, shift, truth[3])
        jacobian = sequential.optimizer_jacobian(
            j_rh, j_rv, latent, sequential.min_ratio, sequential.max_ratio,
            log_rv_cap=sequential.log_rv_cap, log_rh=rh)
        if not np.isfinite(prediction).all() or not np.isfinite(jacobian).all():
            raise ValueError('Non-finite 2D adjoint evaluation')
        return jacobian[mask]

    map_iterations = (args.gated_map_iterations if gate is not None and gate['engaged']
                      else args.max_iterations)
    if args.strategy == 'rml-reset':
        started = time.monotonic()
        print('RML: selecting top 20 1D members; independent randomized priors and full-Cd data targets',
              flush=True)

        def forward_batch(models):
            parameters = [sequential._build_param_dict(physical_model(model))[0] for model in models]
            return adapter.run_pred_multi(parameters, args.index, shift, truth[3])[:, mask]

        def derivative_batch(models):
            packages = [sequential._build_param_dict(physical_model(model)) for model in models]
            evaluated = adapter.run_grad_multi([item[0] for item in packages], args.index, shift, truth[3])
            jacobians = []
            for (_, rh, latent), (prediction, j_rh, j_rv) in zip(packages, evaluated):
                if j_rh is None or j_rv is None:
                    jacobians.append(None)
                else:
                    jacobians.append(sequential.optimizer_jacobian(
                        j_rh, j_rv, latent, sequential.min_ratio, sequential.max_ratio,
                        log_rv_cap=sequential.log_rv_cap, log_rh=rh)[mask])
            return jacobians

        ensemble_fit = run_rml(
            state, predicted_final, observations, covariance, prior_sp,
            forward_batch, derivative_batch, np.random.default_rng(args.seed + 20000 + args.index),
            count=args.rml_members, max_iterations=map_iterations, max_trials=args.max_trials,
            max_seconds=args.max_seconds, max_step=args.max_step,
            relative_data_improvement=args.relative_data_improvement, rh_limits=(lower, upper),
            progress=lambda iteration, active: print(
                f'RML wave {iteration}/{map_iterations}: {active} independent optimizers active',
                flush=True))
        fits = ensemble_fit['fits']
        models = np.column_stack([fit['model'] for fit in fits])
        fitted_predictions = np.column_stack([
            fit['prediction'] if np.shape(fit['prediction']) == observations.shape
            else np.full_like(observations, np.nan) for fit in fits])
        write_npz(output / 'rml-inputs.npz', selected_member_ids=ensemble_fit['selected'],
                  initials=ensemble_fit['initials'], prior_centers=ensemble_fit['prior_centers'],
                  targets=ensemble_fit['targets'], measurement_noise=ensemble_fit['measurement_noise'],
                  observations=observations, data_variance=covariance)
        write_npz(output / 'rml-models.npz', fields=models, predictions=fitted_predictions,
                  usable=ensemble_fit['usable'])
        summaries = [{key: fit[key] for key in
                      ('converged', 'reason', 'objective_history', 'data_misfit_history',
                       'relative_data_improvement_history', 'trial_errors', 'seconds')}
                     for fit in fits]
        result = {'status': 'rml_failed', 'strategy': args.strategy, 'logging_index': args.index,
                  'gate': gate, 'mda_history': history, 'mda_likelihood_weight': 1.,
                  'rml_count': len(fits), 'rml_usable_count': int(ensemble_fit['usable'].sum()),
                  'rml_converged_count': sum(bool(fit['converged']) for fit in fits),
                  'rml_iteration_cap_count': sum(fit['reason'] == 'iteration_limit' for fit in fits),
                  'rml_fits': summaries,
                  'rml_original_data_misfits': [measurement_misfit(
                      fitted_predictions[:, j], observations, covariance)
                      if np.isfinite(fitted_predictions[:, j]).all() else None
                      for j in range(len(fits))]}
        if restart is not None:
            result['restart'] = restart
        if not ensemble_fit['usable'].all():
            (output / 'result.json').write_text(json.dumps(result, indent=2))
            print(f"RML failed: {result['rml_usable_count']}/{len(fits)} usable fits", flush=True)
            return
        raw, posterior_mean, posterior_std = gaussian_moment_redraw(
            models, (nx, nz), (dx, dz),
            ((RH_CORRELATION_X_FT, RH_CORRELATION_Z_FT),
             (manifest['prior_latent_correlation_x_ft'], manifest['prior_latent_correlation_z_ft'])),
            args.ensemble_size, args.seed + 30000 + args.index)
        samples = raw.copy()
        samples[:cells] = np.clip(samples[:cells], lower, upper)
        samples[cells:] = np.clip(samples[cells:], -4., 4.)
        mean_predictions = forward_batch([models.mean(axis=1), samples.mean(axis=1)])
        result.update(status='rml_completed',
                      rml_mean_model_2d_reduced_misfit=measurement_misfit(
                          mean_predictions[0], observations, covariance),
                      final_reduced_misfit=measurement_misfit(mean_predictions[1], observations, covariance),
                      final_rh_rmse_log10=rh_rmse(samples, reference, truth),
                      clipped_parameter_fraction=float(np.mean(samples != raw)),
                      total_seconds=time.monotonic() - started,
                      posterior_sampling='fast_gaussian Rh + exponential latent; exact RML marginal moments before bounds')
        write_npz(output / 'rml-posterior-moments.npz', mean=posterior_mean,
                  variance=posterior_std ** 2, std=posterior_std,
                  achieved_mean=samples.mean(axis=1), achieved_std=samples.std(axis=1, ddof=1))
        write_npz(output / 'gaussian-redraw.npz', fields=samples, raw_fields=raw,
                  member_ids=np.arange(args.ensemble_size))
        write_npz(output / 'final_ensemble.npz', fields=samples,
                  member_ids=np.arange(args.ensemble_size), kind='rml_gaussian_redraw')
        result['curtain'] = render_curtain(
            output / 'curtain-final-rml.png', samples, study_meta, reference, truth_pkg,
            f"Index {args.index} • 20 RML fits → 250 Gaussian draws • "
            f"mean-model misfit {result['final_reduced_misfit']:.4g}")
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        print(f"RML complete: {result['rml_converged_count']}/20 converged, "
              f"{result['rml_iteration_cap_count']} capped; redraw misfit "
              f"{result['final_reduced_misfit']:.4g}", flush=True)
        return
    started = time.monotonic()
    initial_prediction = forward(initial)
    fit = optimize(initial, prior_sp, observations, variance_map, forward, derivative,
                   max_iterations=map_iterations, max_trials=args.max_trials,
                   max_seconds=args.max_seconds, max_step=args.max_step,
                   relative_data_improvement=args.relative_data_improvement,
                   initial_prediction=initial_prediction, rh_limits=(lower, upper),
                   step_weights=None if independent or distance_loc is None else distance_loc.weights)
    stalled = not independent and not fit['converged'] and fit['reason'] == 'no_acceptable_trial'
    cap_converged = (not independent and not fit['converged'] and gate is not None and gate['engaged']
                     and fit['reason'] == 'iteration_limit')
    convergence_kind = ('relative_data_improvement' if fit['converged']
                        else 'iteration_cap' if cap_converged
                        else 'line_search_stall' if stalled else 'none')
    result = {'status': 'incomplete', 'logging_index': args.index,
              'strategy': args.strategy, 'map_data_inflation': map_inflation,
              'mda_likelihood_weight': float(np.reciprocal(inflation).sum()),
              'best_member': best_member,
              'best_member_1d_misfit': float(member_misfits[best_member]),
              'initial_2d_reduced_misfit': measurement_misfit(
                  initial_prediction, observations, covariance),
              'map_converged': bool(fit['converged'] or cap_converged),
              'map_reason': fit['reason'],
              'map_convergence_kind': convergence_kind,
              'map_stalled_accepted': bool(stalled),
              'map_cap_converged': bool(cap_converged),
              'map_accepted_updates': len(fit['objective_history']) - 1,
              'map_reduced_misfit': measurement_misfit(
                  fit['prediction'], observations, covariance),
              'map_rh_rmse_log10': rh_rmse(physical_model(fit['model'])[:, None], reference, truth),
              'objective_history': fit['objective_history'],
              'data_misfit_history': fit['data_misfit_history'],
              'relative_data_improvement_history': fit['relative_data_improvement_history'],
              'trial_errors': fit['trial_errors'], 'map_seconds': fit['seconds'],
              'map_iterations_used': map_iterations,
              'mda_history': history}
    if gate is not None:
        result['gate'] = gate
    if restart is not None:
        result['restart'] = restart
    print(f"MAP: converged={fit['converged']} reason={fit['reason']} "
          f"updates={result['map_accepted_updates']} "
           f"misfit={result['map_reduced_misfit']:.4g}", flush=True)
    write_npz(output / 'map-solution.npz', model=fit['model'], prediction=fit['prediction'],
              jacobian=fit['jacobian'], data_variance=variance_map)
    if not fit['converged'] and not stalled and not cap_converged:
        result['status'] = 'map_not_converged'
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        print('No Laplace posterior drawn: MAP did not converge', flush=True)
        return
    if stalled:
        print('MAP line search stalled; using the last accepted iterate as the MAP', flush=True)
    if cap_converged:
        print('MAP completed the gated iteration cap; flagged converged '
              '(kind=iteration_cap)', flush=True)

    raw = sample_laplace(fit, prior_sp, variance_map, args.ensemble_size,
                          np.random.default_rng(args.seed + 10000))
    samples = raw.copy()
    samples[:cells] = np.clip(samples[:cells], lower, upper)
    samples[cells:] = np.clip(samples[cells:], -4., 4.)
    result['clipped_parameter_fraction'] = float(np.mean(samples != raw))
    if distance_loc is not None and not independent:
        samples = blend_localized_samples(state, samples, distance_loc.weights)
        result['posterior_sampling'] = 'Laplace proposal blended into incoming members with PET fb weights'
        result['protected_collar_max_change'] = float(np.max(
            np.abs(samples[distance_loc.weights == 0] - state[distance_loc.weights == 0])))
        collar_change(samples)
    if independent:
        result['posterior_sampling'] = 'direct independent 2D Laplace draws, bounded to physical limits'
        result['posterior_mean_model_2d_reduced_misfit'] = measurement_misfit(
            forward(samples.mean(axis=1)), observations, covariance)
    pre_rtps = None
    if args.rtps_strength > 0:
        pre_rtps = samples.copy()
        samples, diagnostics = localized_rtps(
            samples, prior_sp.std, distance_loc.weights, args.rtps_strength)
        before_bounds = samples.copy()
        samples[:cells] = np.clip(samples[:cells], lower, upper)
        samples[cells:] = np.clip(samples[cells:], -4., 4.)
        collar_change(samples)
        diagnostics['std_achieved'] = samples.std(axis=1, ddof=1)
        diagnostics['mean_shift_after_bounds'] = samples.mean(axis=1) - pre_rtps.mean(axis=1)
        write_npz(output / 'rtps-diagnostics.npz', **diagnostics)
        result['rtps'] = {
            'strength': args.rtps_strength,
            'reference': 'SpatialPrior.std including floors',
            'inflated_parameter_fraction': float(np.mean(diagnostics['inflation_factors'] > 1)),
            'inflation_factor_max': float(diagnostics['inflation_factors'].max()),
            'zero_spread_unrestored_count': int(diagnostics['zero_spread_unrestored'].sum()),
            'bounds_clipped_fraction': float(np.mean(samples != before_bounds)),
            'max_abs_mean_shift_after_bounds': float(np.max(np.abs(
                diagnostics['mean_shift_after_bounds']))),
            'protected_collar_max_change': 0.}
        print(f"Localized RTPS {args.rtps_strength:g}: "
              f"{result['rtps']['inflated_parameter_fraction']:.1%} parameters inflated; "
              f"max factor {result['rtps']['inflation_factor_max']:.3g}", flush=True)
    sample_arrays = {'fields': samples, 'raw_fields': raw,
                     'member_ids': np.arange(args.ensemble_size)}
    if pre_rtps is not None:
        sample_arrays['pre_rtps_fields'] = pre_rtps
    skip_forecast = bool(gate is not None and gate['engaged'])
    if skip_forecast:
        write_npz(output / 'laplace_samples.npz', **sample_arrays)
        write_npz(output / 'final_ensemble.npz', fields=samples,
                  member_ids=np.arange(args.ensemble_size), kind='laplace_unverified')
        result['status'] = 'complete_map_only'
        result['posterior_forward_valid'] = None
        result['posterior_forward_evaluation'] = 'skipped: engaged gate reports the MAP only'
        result['posterior_ensemble_mean_reduced_misfit'] = None
        result['posterior_mean_rh_rmse_log10'] = rh_rmse(samples, reference, truth)
        result['total_seconds'] = time.monotonic() - started
        result['curtain'] = render_curtain(
            output / 'curtain-final-2d.png', samples, study_meta, reference, truth_pkg,
            f"Index {args.index} • 2D MAP (Laplace draws not forward-verified) • "
            f"MAP misfit {result['map_reduced_misfit']:.4g}")
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        print(f"MAP-only report: misfit {result['map_reduced_misfit']:.4g}, "
              f"RMSE {result['map_rh_rmse_log10']:.4g}; "
              f"{args.ensemble_size} Laplace draws saved unverified", flush=True)
        return
    batch = adapter.run_pred_multi(
        [sequential._build_param_dict(samples[:, i])[0] for i in range(samples.shape[1])],
        args.index, shift, truth[3])
    predictions = np.asarray(batch, dtype=float)[:, mask].T
    valid = np.isfinite(predictions).all(axis=0)
    write_npz(output / 'laplace_samples.npz', **sample_arrays)
    write_npz(output / 'laplace_evaluation.npz', predictions=predictions, valid=valid,
              member_ids=np.arange(args.ensemble_size))
    write_npz(output / 'final_ensemble.npz', fields=samples,
              member_ids=np.arange(args.ensemble_size), kind='laplace_verified')
    result['status'] = 'complete' if valid.any() else 'posterior_forward_failed'
    result['posterior_forward_valid'] = int(valid.sum())
    result['posterior_invalid_member_ids'] = np.flatnonzero(~valid).tolist()
    result['posterior_ensemble_mean_reduced_misfit'] = (
        measurement_misfit(predictions[:, valid].mean(axis=1), observations, covariance)
        if valid.any() else None)
    result['posterior_mean_rh_rmse_log10'] = (
        rh_rmse(samples[:, valid], reference, truth) if valid.any() else None)
    result['total_seconds'] = time.monotonic() - started
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    if valid.any():
        result['curtain'] = render_curtain(
            output / 'curtain-final-2d.png', samples[:, valid], study_meta, reference,
            truth_pkg, f"Index {args.index} • 2D MAP/Laplace posterior • "
                       f"{int(valid.sum())}/{args.ensemble_size} forward-valid • "
                       f"MAP misfit {result['map_reduced_misfit']:.4g}")
        (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(f"Laplace: sampled={args.ensemble_size}, "
          f"2D forward-valid={int(valid.sum())}, "
          f"mean misfit={result['posterior_ensemble_mean_reduced_misfit']}", flush=True)


if __name__ == '__main__':
    main()
