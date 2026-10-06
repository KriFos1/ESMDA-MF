"""Sequential 0D/1D ES-MDA and local-window 2D gradient inversion."""

import argparse
import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import time

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

from esmda_core import (LOG_RH_LIMITS, LocalWindow, MemberEvaluationError, clamp_log_rh,
                        measurement_misfit, noise_floor_mask, observation_vectors,
                        optimizer_jacobian, optimize_members, project_to_modes,
                        shift_posterior, shift_posterior_1d, update_members_once)
from esmda_remote2d import UDAR_COMPONENTS, SimBackendConfig, make_remote2d
import tv_regularization as tv

meter_to_feet = 3.28084
dims = (15, 5, 20)
dX = dY = dZ = 5.0
min_ratio, max_ratio = 1.0, 5.0
tools = ["('6kHz','83ft')", "('12kHz','83ft')", "('24kHz','83ft')",
         "('24kHz','43ft')", "('48kHz','43ft')", "('96kHz','43ft')"]
masked_udar_components = ('USDP', 'USDA')
log_normal_rh_std_target = 3.6
normal_rh_mean = 1.0
ratio_latent_mean = 0.0
ratio_latent_std = 0.25


def setup_simulators(benchmark_dir, tool_list):
    from EMsim.EM import UTA0D, UTA1D
    global UTA0D_sim, UTA1D_sim, wellpath, TVD, MD, WX, WY
    with h5py.File(Path(benchmark_dir) / 'globalmodel.h5', 'r') as handle:
        wellpath = {key: value[:] for key, value in handle['wellpath'].items()}
    TVD, MD, WX, WY = [wellpath[key][:, 0] * meter_to_feet
                       for key in ('Z', 'Distance', 'X', 'Y')]
    tool_path = Path('tool.inp').resolve()
    with tool_path.open('w') as stream:
        stream.write('freq,sub\n')
        for tool in tool_list:
            freq, sub = ast.literal_eval(tool)
            stream.write(f'{int(freq[:-3])},{int(sub[:-2])}\n')
    inputs = {'toolflag': 0, 'datasign': 3, 'anisoflag': 1,
              'toolsetting': str(tool_path),
              'reference_model': str(Path(benchmark_dir) / 'globalmodel.h5'),
              'map': {'ratio': [1, 5]}, 'obsname': 'tvd',
              'assimindex': [[0]], 'datatype': tool_list}
    UTA0D_sim = UTA0D(inputs)
    UTA0D_sim.setup_fwd_run(redund_sim=None)
    UTA1D_sim = UTA1D({**inputs, 'surface_depth': list(np.arange(dims[2]) * dZ)})
    UTA1D_sim.setup_fwd_run(redund_sim=None)


def normal_std_from_lognormal_std(sigma_X, mu):
    def equation(sigma):
        return np.sqrt((np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)) - sigma_X
    return float(fsolve(equation, 1.0)[0])


def sample_prior_0D(ne, background, reservoir, mode_std, fraction):
    """Two-mode mixture prior in ln Rh: background and reservoir levels."""
    reservoirs = np.random.random(ne) < fraction
    rh = np.where(reservoirs,
                  np.random.normal(reservoir, mode_std, ne),
                  np.random.normal(background, mode_std, ne))
    mean = fraction * reservoir + (1. - fraction) * background
    variance = (fraction * (mode_std ** 2 + (reservoir - mean) ** 2)
                + (1. - fraction) * (mode_std ** 2 + (background - mean) ** 2))
    covariance = np.diag([variance, ratio_latent_std ** 2])
    samples = np.vstack([rh, np.random.normal(ratio_latent_mean, ratio_latent_std, ne)])
    return samples, covariance, np.array([mean, ratio_latent_mean])


def sample_prior_0D_gaussian(ne):
    rh_std = normal_std_from_lognormal_std(log_normal_rh_std_target, normal_rh_mean)
    covariance = np.diag([rh_std ** 2, ratio_latent_std ** 2])
    samples = np.vstack([np.random.normal(normal_rh_mean, rh_std, ne),
                         np.random.normal(ratio_latent_mean, ratio_latent_std, ne)])
    return samples, covariance, np.array([normal_rh_mean, ratio_latent_mean])


def mixture_stats(nz, background, reservoir, mode_std, fraction):
    mean = fraction * reservoir + (1. - fraction) * background
    variance = (fraction * (mode_std ** 2 + (reservoir - mean) ** 2)
                + (1. - fraction) * (mode_std ** 2 + (background - mean) ** 2))
    means = np.concatenate([np.full(nz, mean), np.full(nz, ratio_latent_mean)])
    covariance = np.diag(np.concatenate([np.full(nz, variance), np.full(nz, ratio_latent_std ** 2)]))
    return means, covariance


def sample_blocky_prior_1D(ne, zero_d_posterior, background, reservoir, mode_std, fraction,
                         projection=True):
    """Two-mode layered columns with random block edges.

    Each member draws 1-4 switch positions; modes alternate between the
    background and reservoir levels. The segment holding the center cell
    follows the member's 0D posterior side of the level midpoint.
    """
    nz = dims[2]
    midpoint = 0.5 * (background + reservoir)
    rh = np.empty((nz, ne))
    for member in range(ne):
        count = min(int(np.random.randint(1, 5)), nz - 1)
        switches = np.sort(np.random.choice(np.arange(1, nz), size=count, replace=False))
        bounds = np.concatenate(([0], switches, [nz]))
        reservoir_segment = np.random.random() < fraction
        modes = np.empty(nz, dtype=bool)
        for start, stop, state in zip(bounds[:-1], bounds[1:],
                                      [reservoir_segment if i % 2 == 0 else not reservoir_segment
                                       for i in range(len(bounds) - 1)]):
            modes[start:stop] = state
        center_segment = int(np.searchsorted(bounds, nz // 2, side='right')) - 1
        modes[bounds[center_segment]:bounds[center_segment + 1]] = \
            zero_d_posterior[member] > midpoint
        rh[:, member] = np.where(modes,
                                 np.random.normal(reservoir, mode_std, nz),
                                 np.random.normal(background, mode_std, nz))
    if projection:
        rh = project_to_modes(rh, background, reservoir, 2. * mode_std)
    latent = np.random.normal(ratio_latent_mean, ratio_latent_std, (nz, ne))
    means, covariance = mixture_stats(nz, background, reservoir, mode_std, fraction)
    return np.vstack([rh, latent]), covariance, means


def sample_chained_prior_1D(ne, trend, background, reservoir, mode_std, fraction, switch_prob,
                          projection=True):
    """Perturb the shifted previous posterior within its modes, flipping a
    small fraction of cells so newly entering geology stays discoverable."""
    nz = dims[2]
    members = trend.shape[1]
    base = trend[:, np.arange(ne) % members]
    midpoint = 0.5 * (background + reservoir)
    flips = np.random.random((nz, ne)) < switch_prob
    reservoir_cells = base[:nz] > midpoint
    rh = base[:nz] + np.random.normal(0., mode_std, (nz, ne))
    rh = np.where(flips & reservoir_cells,
                  np.random.normal(background, mode_std, (nz, ne)), rh)
    rh = np.where(flips & ~reservoir_cells,
                  np.random.normal(reservoir, mode_std, (nz, ne)), rh)
    if projection:
        rh = project_to_modes(rh, background, reservoir, 2. * mode_std)
    latent = base[nz:] + np.random.normal(0., ratio_latent_std, (nz, ne))
    means, covariance = mixture_stats(nz, background, reservoir, mode_std, fraction)
    return np.vstack([rh, latent]), covariance, means


def resample_prior_0D_to_1D(ne, param_vec_0D):
    from geostat.decomp import Cholesky
    if param_vec_0D.shape != (2, ne):
        raise ValueError('0D ensemble must have shape (2, members)')
    nz = dims[2]
    center = nz // 2
    geostat = Cholesky()
    rh_std = normal_std_from_lognormal_std(log_normal_rh_std_target, param_vec_0D[0].mean())
    samples, covariances, means = [], [], []
    for field, std in enumerate((rh_std, ratio_latent_std)):
        mean = np.full(nz, param_vec_0D[field].mean())
        covariance = geostat.gen_cov2d(nz, 1, std ** 2, nz, 1, 0, 'exp')
        unconditional = geostat.gen_real(mean, covariance, ne).reshape(nz, ne)
        residual = param_vec_0D[field] - unconditional[center]
        conditioned = unconditional + (covariance[:, center] / covariance[center, center])[:, None] * residual
        conditioned[center] = param_vec_0D[field]
        samples.append(conditioned)
        covariances.append(covariance)
        means.append(mean)
    zeros = np.zeros_like(covariances[0])
    return np.vstack(samples), np.block([[covariances[0], zeros], [zeros, covariances[1]]]), np.concatenate(means)


def resample_prior_multiple_1D_to_2D(ne, columns, indices, previous=None, posterior_spread=None):
    """Kriging-conditioned local fields, optionally around an aligned posterior.

    ``posterior_spread`` scales the residual (and returned covariance) per
    field by ``clip(spread/prior_std, 0.1, 1)`` so the 2D prior reflects the
    lower-fidelity posterior uncertainty instead of the full initial spread.
    """
    from geostat.decomp import Cholesky
    nx, _, nz = dims
    n_cells = nx * nz
    indices = np.asarray(indices, dtype=int)
    if not len(indices) or len(columns) != len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError('Conditioning columns must have distinct indices')
    if (indices < 0).any() or (indices >= nx).any():
        raise ValueError('Conditioning column lies outside the window')
    if any(np.shape(column) != (2 * nz, ne) for column in columns):
        raise ValueError('1D columns do not match the window or ensemble')
    if previous is not None and np.shape(previous) != (ne, 2 * n_cells):
        raise ValueError('Previous posterior does not match the local window')
    conditioning = np.concatenate([index * nz + np.arange(nz) for index in indices])
    geostat = Cholesky()
    var_range_cells = max(1, int(np.floor(5.0 * meter_to_feet / dX)))
    vertical_aspect = 4 * dZ / dX
    fields, covariances, means = [], [], []
    for field in range(2):
        values = np.concatenate([column[field*nz:(field+1)*nz] for column in columns])
        if previous is None:
            trend = np.tile(columns[-1][field*nz:(field+1)*nz], (nx, 1))
        else:
            trend = np.asarray(previous).reshape(ne, 2, n_cells)[:, field, :].T.copy()
        trend[conditioning] = values
        mean = trend.mean(axis=1)
        std = (normal_std_from_lognormal_std(log_normal_rh_std_target, min(3, mean.max()))
               if field == 0 else ratio_latent_std)
        if posterior_spread is not None:
            scale = float(np.clip(posterior_spread[field] / std, 0.03, 1.0))
            std *= scale
        covariance = geostat.gen_cov2d(nx, nz, std ** 2, var_range_cells, vertical_aspect, 0, 'exp')
        residual = geostat.gen_real(np.zeros(n_cells), covariance, ne).reshape(n_cells, ne)
        C_dd = covariance[np.ix_(conditioning, conditioning)] + np.eye(len(conditioning)) * np.finfo(float).eps
        weights = np.linalg.solve(C_dd, -residual[conditioning])
        residual += covariance[:, conditioning] @ weights
        residual[conditioning] = 0
        conditioned = trend + residual
        conditioned[conditioning] = values
        fields.append(conditioned)
        covariances.append(covariance)
        means.append(mean)
    zeros = np.zeros_like(covariances[0])
    return np.vstack(fields), np.block([[covariances[0], zeros], [zeros, covariances[1]]]), np.concatenate(means)


def _build_param_dict(param):
    n_cells = dims[0] * dims[2]
    param = np.asarray(param, dtype=float)
    if param.shape != (2 * n_cells,):
        raise ValueError('2D parameter vector does not match the local window')
    log_rh = param[:n_cells].reshape(dims[0], dims[2])
    latent = param[n_cells:].reshape(dims[0], dims[2])
    log_rv = log_rh + np.log(min_ratio) + np.log(max_ratio / min_ratio) * norm.cdf(latent)
    fields = [np.repeat(field[:, None, :], dims[1], axis=1).ravel(order='F')
              for field in (log_rh, log_rv)]
    return dict(zip(('rh', 'rv'), fields)), log_rh, latent


def _set_logging_point_state(logging_point_idx):
    for sim in (UTA0D_sim, UTA1D_sim):
        for key, source in (('tvd', 'Z'), ('MD', 'Distance'), ('X', 'X'), ('ijk', 'Y')):
            sim.tool[key] = wellpath[source][logging_point_idx:logging_point_idx+1, 0] * meter_to_feet
        for key, source in (('dip', 'Inclination'), ('Azim', 'Azimuth')):
            sim.tool[key] = wellpath[source][logging_point_idx:logging_point_idx+1, 0]
        sim.tool['degwell'] = [np.array([sim.tool['dip'][0], sim.tool['Azim'][0]])]
        sim.tool['nlog'] = 1
    UTA1D_sim.tool['surface_depth'] = [current_shift['z'] + i * dZ for i in range(dims[2])]


def component_positions(n_channels):
    """Channels of the ultra-shallow UDAR components, sub-cell for the 2D grid."""
    width = len(UDAR_COMPONENTS)
    if n_channels % width:
        raise ValueError('Channel count must be a multiple of the UDAR component count')
    positions = []
    for name in masked_udar_components:
        positions.extend(range(UDAR_COMPONENTS.index(name), n_channels, width))
    return np.asarray(sorted(positions), dtype=int)


def soften_column_edges(column, nz):
    """Turn a blocky log-Rh column into a two-cell ramp (5 ft cell average)."""
    softened = np.asarray(column, dtype=float).copy()
    rh = softened[:nz]
    padded = np.r_[rh[0], rh, rh[-1]]
    softened[:nz] = np.convolve(padded, [.25, .5, .25], mode='valid')
    return softened


def simulate_pred_only(param, logging_point_idx=0, simfidelity='2D'):
    if simfidelity == '2D':
        parameters, _, _ = _build_param_dict(param)
        return remote_2d.run_pred(parameters, logging_point_idx, current_shift, wellpath)
    if simfidelity not in ('0D', '1D'):
        raise ValueError(f'Unknown fidelity {simfidelity}')
    _set_logging_point_state(logging_point_idx)
    size = 1 if simfidelity == '0D' else dims[2]
    log_rh = np.asarray(param[:size])
    latent = np.asarray(param[size:])
    log_rv = log_rh + np.log(min_ratio) + np.log(max_ratio / min_ratio) * norm.cdf(latent)
    parameters = {'rh': log_rh.item() if size == 1 else log_rh,
                  'rv': log_rv.item() if size == 1 else log_rv}
    sim = UTA0D_sim if simfidelity == '0D' else UTA1D_sim
    prediction = sim.run_fwd_sim(parameters, 0)
    vector = np.concatenate([np.asarray(prediction[0][tool]).ravel() for tool in tools])
    if vector.shape != (8 * len(tools),) or not np.isfinite(vector).all():
        raise RuntimeError(f'Invalid {simfidelity} UDAR response')
    return vector


def simulate_and_grad(param, logging_point_idx=0):
    parameters, _, latent = _build_param_dict(param)
    prediction, j_rh, j_rv = remote_2d.run_grad(parameters, logging_point_idx, current_shift, wellpath)
    return prediction, optimizer_jacobian(j_rh, j_rv, latent, min_ratio, max_ratio)


def simulate_and_grad_multi(param_list, logging_point_idx=0):
    """Batched 2D evaluation; one (prediction, jacobian) or Exception per input."""
    results = remote_2d.run_grad_multi([_build_param_dict(param)[0] for param in param_list],
                                       logging_point_idx, current_shift, wellpath)
    outcome = []
    for param, (prediction, j_rh, j_rv) in zip(param_list, results):
        if j_rh is None:
            outcome.append(RuntimeError('2D evaluation failed for one member field'))
            continue
        _, _, latent = _build_param_dict(param)
        outcome.append((prediction, optimizer_jacobian(j_rh, j_rv, latent, min_ratio, max_ratio)))
    return outcome


def custom_loss(predictions, data_real, theta, theta_mean, C_theta, Cd):
    residual = np.asarray(data_real).ravel() - np.asarray(predictions).ravel()
    prior_residual = np.asarray(theta).ravel() - np.asarray(theta_mean).ravel()
    return 0.5 * (np.sum(residual ** 2 / Cd) + prior_residual @ np.linalg.solve(C_theta, prior_residual))


def map_update_data_space_LM(theta, f_theta, theta_mean, J, y, Cd_diag, C_theta, step_lambda):
    scale = 1 + step_lambda
    prior_residual = (np.asarray(theta).ravel() - np.asarray(theta_mean).ravel()) / scale
    CM_JT = C_theta @ J.T
    H_data = scale * np.diag(Cd_diag) + J @ CM_JT
    rhs = np.asarray(f_theta).ravel() - np.asarray(y).ravel() - J @ prior_residual
    try:
        weights = np.linalg.solve(H_data, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(H_data, rhs, rcond=None)[0]
    return -prior_residual - CM_JT @ weights


def parse_args(argv=None):
    config = SimBackendConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    config.add_cli(parser)
    here = Path(__file__).resolve().parent
    parser.add_argument('--benchmark', type=Path, default=here.parent / 'Dashboard/benchmarks/Benchmark-3')
    parser.add_argument('--data', type=Path, default=here / 'data.pkl')
    parser.add_argument('--variance', type=Path, default=here / 'var.pkl')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh directory for outputs and simulator files')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--frequency', type=int, default=1)
    parser.add_argument('--nx', type=int, default=15)
    parser.add_argument('--nz', type=int, default=20)
    parser.add_argument('--y-layers', type=int, default=5)
    parser.add_argument('--dx-ft', type=float, default=5.)
    parser.add_argument('--dz-ft', type=float, default=5.)
    parser.add_argument('--ensemble-size', type=int, default=250)
    parser.add_argument('--opt-members', type=int, default=5)
    parser.add_argument('--max-iterations', type=int, default=5)
    parser.add_argument('--max-substitutions', type=int, default=8,
                        help='Per-point 2D member substitutions before escalation is abandoned '
                             'in favor of the final 1D update')
    parser.add_argument('--max-lm-trials', type=int, default=2)
    parser.add_argument('--misfit-threshold', type=float, default=4.,
                        help='Reduced measurement misfit below which 1D is deemed sufficient')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--force-2d', action='store_true', help='Exercise 2D even if the lower fidelities converge')
    parser.add_argument('--single-update', action='store_true',
                        help='Engaged 2D points take one batched linearized update '
                             'instead of iterated LM (one solver batch per point)')
    parser.add_argument('--prior-style', choices=['blocky', 'gaussian'], default='blocky',
                        help='1D prior family: two-mode blocky columns or the legacy smooth Gaussian')
    parser.add_argument('--prior-background', type=float, default=float(np.log10(1.5)),
                        help='log10 Rh of the background mode (Ohm-m)')
    parser.add_argument('--prior-reservoir', type=float, default=float(np.log10(30.)),
                        help='log10 Rh of the reservoir mode (Ohm-m)')
    parser.add_argument('--prior-mode-std', type=float, default=0.10,
                        help='Within-mode log10 Rh standard deviation')
    parser.add_argument('--prior-reservoir-fraction', type=float, default=0.15,
                        help='Fraction of reservoir-mode cells in the prior')
    parser.add_argument('--prior-switch-prob', type=float, default=0.05,
                        help='Per-cell mode-flip probability when chaining on the previous posterior')
    parser.add_argument('--prior-projection', choices=['on', 'off'], default='on',
                        help='Mode projection for the blocky prior; off retains identical chaining')
    parser.add_argument('--regularization', choices=['none', 'tv'], default='none',
                        help='Approximate TV-regularized ensemble updates (disables mode projection)')
    parser.add_argument('--tv-weight', type=float, default=.05,
                        help='Total per-point TV dose in natural-log coordinates')
    parser.add_argument('--tv-weight-1d', type=float, default=None,
                        help='1D TV coefficient; defaults to --tv-weight')
    parser.add_argument('--tv-weight-2d', type=float, default=None,
                        help='2D TV coefficient; defaults to --tv-weight')
    parser.add_argument('--tv-epsilon', type=float, default=1e-3,
                        help='Smoothed-TV gradient scale in ln resistivity per foot')
    parser.add_argument('--tv-iterations', type=int, default=8,
                        help='Maximum local IRLS sweeps; no extra simulator calls')
    parser.add_argument('--tv-max-step', type=float, default=.5,
                        help='Maximum active 2D increment in optimizer coordinates')
    parser.add_argument('--tv-amplitude-std', type=float, default=float(np.log(10.)),
                        help='Weak independent 2D ln Rh prior standard deviation')
    args = parser.parse_args(argv)
    if args.ensemble_size < 2 or not 1 <= args.opt_members <= args.ensemble_size:
        parser.error('Require ensemble-size >= 2 and 1 <= opt-members <= ensemble-size')
    if min(args.frequency, args.nx, args.nz, args.y_layers, args.max_lm_trials) < 1 or args.max_iterations < 0:
        parser.error('Sizes, frequency and LM trials must be positive; iterations must be nonnegative')
    if args.max_substitutions < 0:
        parser.error('Require max-substitutions >= 0')
    if not np.isfinite([args.dx_ft, args.dz_ft]).all() or min(args.dx_ft, args.dz_ft) <= 0:
        parser.error('Cell sizes must be positive and finite')
    if not np.isfinite(args.misfit_threshold) or args.misfit_threshold <= 0:
        parser.error('Misfit threshold must be positive and finite')
    if args.start < 0 or (args.stop is not None and args.stop <= args.start):
        parser.error('Require 0 <= start < stop')
    if not np.isfinite([args.prior_background, args.prior_reservoir, args.prior_mode_std,
                        args.prior_reservoir_fraction, args.prior_switch_prob]).all():
        parser.error('Prior parameters must be finite')
    if not args.prior_background < args.prior_reservoir:
        parser.error('Require prior-background < prior-reservoir (log10 Ohm-m)')
    if min(args.prior_mode_std, args.prior_reservoir_fraction) <= 0 \
            or args.prior_reservoir_fraction >= 1 or not 0 <= args.prior_switch_prob < 1:
        parser.error('Require 0 < mode-std, 0 < reservoir-fraction < 1, 0 <= switch-prob < 1')
    if not np.isfinite([args.tv_weight, args.tv_epsilon, args.tv_max_step, args.tv_amplitude_std]).all() \
            or args.tv_weight < 0 or min(args.tv_epsilon, args.tv_max_step, args.tv_amplitude_std) <= 0 \
            or args.tv_iterations < 1:
        parser.error('Require finite nonnegative TV weight and positive TV scales/iterations')
    for weight in (args.tv_weight_1d, args.tv_weight_2d):
        if weight is not None and (not np.isfinite(weight) or weight < 0):
            parser.error('Stage-specific TV coefficients must be finite and nonnegative')
    if args.regularization == 'tv' and not args.single_update:
        parser.error('TV currently uses --single-update with fixed-Jacobian inner IRLS')
    n_cells = args.nx * args.nz
    if (2 * n_cells) ** 2 * 8 > 1024 ** 3:
        parser.error('Local-window dense covariance exceeds 1 GiB; use a smaller window')
    return args, config.from_args(args)


def prior_parameters_ln(args):
    """Convert the documented log10 CLI levels and width once at the boundary."""
    return tuple(np.log(10.) * np.array([args.prior_background, args.prior_reservoir,
                                       args.prior_mode_std]))


def main(argv=None):
    global dims, dX, dY, dZ, tools, current_shift, remote_2d
    args, config = parse_args(argv)
    tv_weight_1d = args.tv_weight if args.tv_weight_1d is None else args.tv_weight_1d
    tv_weight_2d = args.tv_weight if args.tv_weight_2d is None else args.tv_weight_2d
    background, reservoir, mode_std = prior_parameters_ln(args)
    projection = (args.prior_style == 'blocky' and args.prior_projection == 'on'
                  and args.regularization != 'tv')
    from pipt.update_schemes.update_methods_ns.approx_update import approx_update
    benchmark = args.benchmark.resolve()
    data = pd.read_pickle(args.data.resolve())
    variances = pd.read_pickle(args.variance.resolve())
    if not data.index.equals(variances.index) or not data.columns.equals(variances.columns):
        raise ValueError('Data and variance indices/tools do not match')
    with h5py.File(benchmark / 'globalmodel.h5', 'r') as handle:
        depths = handle['wellpath/Z'][:, 0] * meter_to_feet
    if len(data) != len(depths) or not np.allclose(data.index.to_numpy(), depths):
        raise ValueError('Observation rows do not match the benchmark trajectory')
    stop = len(data) if args.stop is None else args.stop
    if not args.start < stop <= len(data):
        raise ValueError('Assimilation range lies outside the benchmark')
    tool_pairs = list(data.columns)
    tools = [str(tool) for tool in tool_pairs]
    for index in range(args.start, stop, args.frequency):
        observation_vectors(data, variances, index, tool_pairs)
    dims = (args.nx, args.y_layers, args.nz)
    dX, dY, dZ = args.dx_ft, args.dx_ft, args.dz_ft
    window = LocalWindow(dims, (dX, dY, dZ))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_cwd = Path.cwd()
    np.random.seed(args.seed)
    try:
        os.chdir(output)
        setup_simulators(benchmark, tools)
        remote_2d = make_remote2d(config, dims, dX, dY, dZ, tools)
        remote_2d.jobs_dir = str(output / 'jobs')
        if not remote_2d.backend().trusts_ad('2D'):
            raise RuntimeError('Select an exact-adjoint 2D solver build')
        manifest = {**vars(args), 'benchmark': benchmark, 'simulator': asdict(config),
                    'runtime': remote_2d.runtime_identity(),
                    'tools': tool_pairs, 'dims': dims, 'spacing_ft': (dX, dY, dZ),
                     'covariance_bytes': (2 * args.nx * args.nz) ** 2 * 8,
                     'parameter_units': '[ln(Rh / Ohm m), anisotropy latent]',
                     'prior_projection_effective': projection,
                     'tv_weights_effective': {'1D': tv_weight_1d, '2D': tv_weight_2d},
                     'prior_units': {'cli': 'log10', 'internal': 'natural_log',
                                     'background_ohm_m': float(np.exp(background)),
                                     'reservoir_ohm_m': float(np.exp(reservoir)),
                                     'background_ln': background, 'reservoir_ln': reservoir,
                                     'mode_std_ln': mode_std, 'projection_radius_ln': 2 * mode_std}}
        (output / 'run.json').write_text(json.dumps(manifest, default=str, indent=2))
        print(f'Local window {dims}, spacing {(dX, dY, dZ)} ft; backend={config.backend}', flush=True)
        previous = None
        previous_kind = None
        previous_1d = None
        previous_index = None
        for index in range(args.start, stop, args.frequency):
            point_started = time.monotonic()
            current_shift, movement = window.update((WX[index], WY[index], TVD[index]))
            _set_logging_point_state(index)
            aligned = (shift_posterior(previous, dims, movement)
                       if previous_kind == '2D' else None)
            observations, Cd = observation_vectors(data, variances, index, tool_pairs)
            sub_cell = np.zeros(len(observations), dtype=bool)
            sub_cell[component_positions(len(observations))] = True
            obs_mask = ~(noise_floor_mask(observations, Cd) | sub_cell)
            if not obs_mask.any():
                raise ValueError(f'Index {index}: every channel is masked')
            masked_channels = int((~obs_mask).sum())
            if masked_channels:
                print(f'Index {index}: masking {masked_channels} unrepresentable channels', flush=True)
            observations, Cd = observations[obs_mask], Cd[obs_mask]

            def predict_ensemble(ens, fidelity):
                full = np.column_stack([simulate_pred_only(ens[:, member], index, fidelity)
                                        for member in range(ens.shape[1])])
                return full[obs_mask]

            threshold = args.misfit_threshold
            if args.prior_style == 'gaussian':
                ensemble, covariance, mean = sample_prior_0D_gaussian(args.ensemble_size)
            else:
                ensemble, covariance, mean = sample_prior_0D(
                    args.ensemble_size, background, reservoir,
                    mode_std, args.prior_reservoir_fraction)
            ensemble = clamp_log_rh(ensemble)
            updater = approx_update()
            ne = args.ensemble_size
            updater.proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
            updater.trunc_energy, updater.keys_da, updater.lam = .95, {}, 0
            updater.list_states = ['rh', 'rv']
            alpha, used_weight = 10., 0.
            misfit = {}
            tv_history = []
            spatial_weight = None
            for fidelity, budget in (('0D', .25), ('1D', .75)):
                if fidelity == '1D':
                    spatial_weight = 1. - used_weight
                    if args.prior_style == 'gaussian':
                        ensemble, covariance, mean = resample_prior_0D_to_1D(ne, ensemble)
                    elif previous_1d is not None:
                        trend = shift_posterior_1d(previous_1d.T, dims[2], movement).T
                        ensemble, covariance, mean = sample_chained_prior_1D(
                            ne, trend, background, reservoir,
                            mode_std, args.prior_reservoir_fraction,
                            args.prior_switch_prob, projection=projection)
                    else:
                        ensemble, covariance, mean = sample_blocky_prior_1D(
                            ne, ensemble[0], background, reservoir,
                            mode_std, args.prior_reservoir_fraction, projection=projection)
                    ensemble = clamp_log_rh(ensemble)
                updater.state_scaling = np.ones(ensemble.shape[0])
                active = ensemble.shape[0] // 2
                updater.prior_info = {key: {'active': active} for key in ('rh', 'rv')}
                prediction = predict_ensemble(ensemble, fidelity)
                misfit_now = measurement_misfit(prediction.mean(axis=1), observations, Cd)
                misfit[f'{fidelity}_initial'] = misfit_now
                while misfit_now > threshold and used_weight < budget:
                    inflated = Cd * alpha
                    updater.scale_data = np.sqrt(inflated)
                    perturbed = np.random.normal(observations[:, None], np.sqrt(inflated)[:, None], (len(Cd), ne))
                    updater.update(enX=ensemble, enY=prediction, enE=perturbed)
                    ensemble += updater.step
                    np.clip(ensemble[:active], *LOG_RH_LIMITS, out=ensemble[:active])
                    if projection:
                        ensemble[:active] = project_to_modes(
                            ensemble[:active], background, reservoir, 2. * mode_std)
                    if args.regularization == 'tv' and fidelity == '1D':
                        dose = tv_weight_1d / alpha / spatial_weight
                        ensemble = tv.regularize_columns(ensemble, dose, dZ,
                                                         args.tv_epsilon, args.tv_iterations)
                        tv_history.append({'stage': '1D', 'strength': dose,
                                           'assimilation_fraction': 1 / alpha / spatial_weight})
                    used_weight += 1 / alpha
                    alpha *= 10 / 9
                    prediction = predict_ensemble(ensemble, fidelity)
                    misfit_now = measurement_misfit(prediction.mean(axis=1), observations, Cd)
                    print(f'Index {index}, {fidelity}: misfit={misfit_now:.6g}, weight={used_weight:.4f}', flush=True)
                misfit[f'{fidelity}_final'] = misfit_now
            ranking_full = np.argsort([custom_loss(prediction[:, member], observations, ensemble[:, member], mean, covariance, Cd)
                                       for member in range(ne)])
            ranking = ranking_full[:args.opt_members]
            selected = ensemble[:, ranking]
            factor = 1 / (1 - used_weight)
            inflated = Cd * factor
            final_tv_fraction = 1 / factor / spatial_weight
            final_tv_dose = tv_weight_2d * final_tv_fraction
            escalate = args.force_2d or misfit_now > threshold
            two_d_evaluations = [0]
            nx, _, nz = dims
            escalation_abandoned = None
            solver_failures = []
            mean_1d, covariance_1d = mean, covariance
            tv_history_mark = len(tv_history)
            if escalate:
                indices = np.arange(nx // 2 + 1)
                if args.prior_style == 'blocky':
                    prior_spread = (mode_std, ratio_latent_std)
                    if aligned is not None and projection:
                        aligned = aligned.copy()
                        aligned[:, :nx * nz] = project_to_modes(
                            aligned[:, :nx * nz], background, reservoir, 2. * mode_std)
                elif aligned is None:
                    half = ensemble.shape[0] // 2
                    prior_spread = (float(np.median(ensemble[:half].std(axis=1))),
                                    float(np.median(ensemble[half:].std(axis=1))))
                else:
                    prior_spread = (float(np.median(aligned[:, :nx * nz].std(axis=0))),
                                    float(np.median(aligned[:, nx * nz:].std(axis=0))))
                columns = []
                for column in indices:
                    if aligned is None or column == nx // 2:
                        center = selected
                    else:
                        center = aligned.reshape(args.opt_members, 2, nx, nz)[:, :, column, :] \
                            .reshape(args.opt_members, 2 * nz).T.copy()
                    if args.prior_style == 'blocky':
                        center = np.apply_along_axis(soften_column_edges, 0, center, nz)
                    columns.append(center.copy())
                prior, covariance, mean = resample_prior_multiple_1D_to_2D(
                    args.opt_members, columns, indices, aligned, prior_spread)
                prior = clamp_log_rh(prior)
                perturbed = np.random.normal(observations[:, None], np.sqrt(inflated)[:, None],
                                             (len(Cd), args.opt_members))
                active_cells = np.ones((2, nx, nz), dtype=bool)
                active_cells[:, :nx // 2] = False

                def evaluate_multi_2d(param_list):
                    two_d_evaluations[0] += len(param_list)
                    results = simulate_and_grad_multi(param_list, index)
                    masked = []
                    for res in results:
                        if isinstance(res, Exception):
                            masked.append(res)
                        else:
                            masked.append((res[0][obs_mask], res[1][obs_mask]))
                    return masked

                def substitute_prior(slot, candidate):
                    center = ensemble[:, [candidate]].copy()
                    if args.prior_style == 'blocky':
                        center = np.apply_along_axis(soften_column_edges, 0, center, nz)
                    member_columns = []
                    for column in indices:
                        if aligned is None or column == nx // 2:
                            member_columns.append(center)
                        else:
                            member_columns.append(aligned.reshape(-1, 2, nx, nz)[slot, :, column, :]
                                                   .reshape(2 * nz, 1))
                    previous = None if aligned is None else aligned.reshape(-1, 2 * nx * nz)[slot:slot + 1]
                    single, _, _ = resample_prior_multiple_1D_to_2D(1, member_columns, indices,
                                                                    previous, prior_spread)
                    return clamp_log_rh(single[:, 0])

                member_candidates = [int(rank) for rank in ranking]
                next_rank = [args.opt_members]

                def substitute(slot, exc):
                    if len(solver_failures) >= args.max_substitutions:
                        print(f'Index {index}, 2D slot {slot}: substitution cap '
                              f'{args.max_substitutions} reached; abandoning 2D escalation', flush=True)
                        return None
                    if next_rank[0] >= ne:
                        print(f'Index {index}, 2D slot {slot}: {exc}; no members left to substitute',
                              flush=True)
                        return None
                    candidate = int(ranking_full[next_rank[0]])
                    next_rank[0] += 1
                    solver_failures.append({'slot': slot, 'member': member_candidates[slot],
                                            'substitute': candidate, 'error': str(exc)})
                    print(f'Index {index}, 2D slot {slot}: initial evaluation failed; '
                          f'substituting ensemble member {candidate}', flush=True)
                    prior[:, slot] = substitute_prior(slot, candidate)
                    perturbed[:, slot] = np.random.normal(observations, np.sqrt(inflated))
                    member_candidates[slot] = candidate
                    return prior[:, slot], perturbed[:, slot]

                try:
                    if args.single_update:
                        update = map_update_data_space_LM
                        if args.regularization == 'tv':
                            def update(theta, prediction, prior_mean, jacobian, datum, data_cov, prior_cov, damping):
                                return tv.linearized_step(
                                    theta, prediction, prior_mean, jacobian, datum, data_cov,
                                    (nx, nz), (dX, dZ), final_tv_dose, active_cells.ravel(),
                                    args.tv_epsilon, args.tv_iterations, args.tv_amplitude_std,
                                    args.tv_max_step, LOG_RH_LIMITS)

                            tv_history.append({'stage': '2D', 'strength': final_tv_dose,
                                               'assimilation_fraction': 1 / factor / spatial_weight})
                        records = update_members_once([prior[:, slot] for slot in range(args.opt_members)],
                                                       [perturbed[:, slot] for slot in range(args.opt_members)],
                                                       mean, covariance, inflated, evaluate_multi_2d,
                                                        update, custom_loss,
                                                       active_cells.ravel(), substitute=substitute)
                    else:
                        records = optimize_members([prior[:, slot] for slot in range(args.opt_members)],
                                                   [perturbed[:, slot] for slot in range(args.opt_members)],
                                                   mean, covariance, inflated, evaluate_multi_2d,
                                                   map_update_data_space_LM, custom_loss,
                                                   args.max_iterations, args.max_lm_trials,
                                                   active_cells.ravel(), substitute=substitute)
                except MemberEvaluationError as exc:
                    escalation_abandoned = str(exc)
                    print(f'Index {index}: 2D escalation abandoned after '
                          f'{len(solver_failures)} substitutions; applying final 1D update', flush=True)
                    mean, covariance = mean_1d, covariance_1d
                    del tv_history[tv_history_mark:]
                    escalate = False
                else:
                    for slot, record in enumerate(records):
                        label = 'gaussian_diagnostic_loss' if args.regularization == 'tv' else 'loss'
                        print(f'Index {index}, 2D member {slot+1} (candidate {member_candidates[slot]}): '
                              f'{label}={record["loss"]:.6g}, iterations={len(record["history"]) - 1}', flush=True)
                    previous = np.stack([record['parameters'] for record in records])
                    previous_kind = '2D'
                    previous_1d = np.stack([np.concatenate(
                        [np.asarray(record['parameters'])[:nx * nz].reshape(nx, nz)[nx // 2, :],
                         np.asarray(record['parameters'])[nx * nz:].reshape(nx, nz)[nx // 2, :]])
                        for record in records], axis=-1)
                    posterior_misfit = [measurement_misfit(record['prediction'], observations, Cd)
                                        for record in records]
                    misfit['before_final'] = misfit['1D_final']
                    misfit['after_final'] = float(np.mean(posterior_misfit))
                    worsened = False
                    results = {'prior_params': prior, 'post_mda_params': prior.copy(),
                               'post_1d_params': selected,
                               'posterior_params': [record['parameters'] for record in records],
                               'posterior_predictions': [record['prediction'] for record in records],
                               'posterior_losses': [record['loss'] for record in records],
                               'optimizer_history': [record['history'] for record in records],
                               'trial_errors': [record['trial_errors'] for record in records],
                               'observations': observations, 'perturbed_observations': perturbed,
                               'data_covariance': inflated, 'prior_mean': mean, 'prior_covariance': covariance,
                               'dims': dims, 'spacing_ft': (dX, dY, dZ), 'shift_ft': dict(current_shift),
                               'window_movement': movement, 'previous_index': previous_index,
                               'logging_index': index, 'tools': tool_pairs, 'seed': args.seed,
                               'mda_weight': used_weight + 1 / factor, 'lower_fidelity_weight': used_weight,
                               'final_inflation_factor': factor,
                               'optimized_2d': True, 'selected_fidelity': '2D', 'two_d_calls': two_d_evaluations[0],
                               'two_d_mode': 'single-update' if args.single_update else 'lm',
                               'prior_style': args.prior_style,
                               'misfit': misfit, 'misfit_threshold': threshold,
                               'posterior_misfit': posterior_misfit, 'final_iteration_worsened': worsened,
                               'member_candidates': member_candidates, 'solver_failures': solver_failures,
                               'substituted_members': len(solver_failures), 'prior_spread': prior_spread,
                               'observation_mask': obs_mask, 'masked_channels': masked_channels,
                               'masked_components': masked_udar_components}
            if not escalate:
                misfit['before_final'] = misfit['1D_final']
                updater.state_scaling = np.ones(ensemble.shape[0])
                active = ensemble.shape[0] // 2
                updater.prior_info = {key: {'active': active} for key in ('rh', 'rv')}
                updater.scale_data = np.sqrt(inflated)
                perturbed = np.random.normal(observations[:, None], np.sqrt(inflated)[:, None], (len(Cd), ne))
                updater.update(enX=ensemble, enY=prediction, enE=perturbed)
                ensemble += updater.step
                np.clip(ensemble[:active], *LOG_RH_LIMITS, out=ensemble[:active])
                if projection:
                    ensemble[:active] = project_to_modes(
                        ensemble[:active], background, reservoir, 2. * mode_std)
                if args.regularization == 'tv':
                    final_tv_dose = tv_weight_1d * final_tv_fraction
                    ensemble = tv.regularize_columns(ensemble, final_tv_dose, dZ,
                                                     args.tv_epsilon, args.tv_iterations)
                    tv_history.append({'stage': 'final_1D', 'strength': final_tv_dose,
                                       'assimilation_fraction': 1 / factor / spatial_weight})
                used_weight += 1 / factor
                prediction = predict_ensemble(ensemble, '1D')
                misfit_now = measurement_misfit(prediction.mean(axis=1), observations, Cd)
                misfit['after_final'] = misfit_now
                worsened = bool(misfit_now > misfit['before_final'])
                print(f'Index {index}, final 1D update: misfit {misfit["before_final"]:.6g} -> '
                      f'{misfit_now:.6g}{" (increased)" if worsened else ""}, weight=1.0000', flush=True)
                ranking = np.argsort([custom_loss(prediction[:, member], perturbed[:, member],
                                                  ensemble[:, member], mean, covariance, inflated)
                                      for member in range(ne)])[:args.opt_members]
                selected = ensemble[:, ranking]
                perturbed = perturbed[:, ranking]
                records = [{'parameters': selected[:, member].copy(),
                            'prediction': prediction[:, ranking[member]],
                            'loss': custom_loss(prediction[:, ranking[member]], perturbed[:, member],
                                                selected[:, member], mean, covariance, inflated)}
                           for member in range(args.opt_members)]
                previous = selected.T
                previous_kind = '1D'
                previous_1d = selected
                posterior_misfit = [measurement_misfit(record['prediction'], observations, Cd)
                                    for record in records]
                results = {'prior_params': None, 'post_mda_params': selected.copy(),
                           'post_1d_params': selected.copy(),
                           'posterior_params': [record['parameters'] for record in records],
                           'posterior_predictions': [record['prediction'] for record in records],
                           'posterior_losses': [record['loss'] for record in records],
                           'optimizer_history': [[record['loss']] for record in records],
                           'trial_errors': [[] for _ in records],
                           'observations': observations, 'perturbed_observations': perturbed,
                           'data_covariance': inflated, 'prior_mean': mean, 'prior_covariance': covariance,
                           'dims': dims, 'spacing_ft': (dX, dY, dZ), 'shift_ft': dict(current_shift),
                           'window_movement': movement, 'previous_index': previous_index,
                           'logging_index': index, 'tools': tool_pairs, 'seed': args.seed,
                           'mda_weight': used_weight, 'lower_fidelity_weight': used_weight - 1 / factor,
                           'final_inflation_factor': factor,
                            'optimized_2d': False, 'selected_fidelity': '1D', 'two_d_calls': two_d_evaluations[0],
                            'prior_style': args.prior_style,
                           'misfit': misfit, 'misfit_threshold': threshold,
                           'posterior_misfit': posterior_misfit, 'final_iteration_worsened': worsened,
                           'member_candidates': [int(rank) for rank in ranking], 'solver_failures': [],
                           'substituted_members': 0,
                            'observation_mask': obs_mask, 'masked_channels': masked_channels,
                            'masked_components': masked_udar_components}
                if escalation_abandoned is not None:
                    results.update(solver_failures=solver_failures,
                                   substituted_members=len(solver_failures),
                                   escalation_abandoned=escalation_abandoned,
                                   two_d_mode='abandoned')
            linearized = escalate and args.single_update
            kind = 'linearized' if linearized else 'forward'
            results.update(parameter_units='natural_log', prior_projection=args.prior_projection,
                           prior_units=manifest['prior_units'], posterior_prediction_kind=kind,
                           posterior_misfit_kind=kind, posterior_forward_verified=not linearized,
                           point_elapsed_seconds=time.monotonic() - point_started)
            results.update(regularization=args.regularization, tv_weight=args.tv_weight,
                           tv_epsilon=args.tv_epsilon, tv_iterations=args.tv_iterations,
                           tv_history=tv_history, prior_projection_effective=projection,
                           tv_strength_applied=sum(entry['strength'] for entry in tv_history),
                           loss_kind='gaussian_diagnostic' if args.regularization == 'tv' else 'gaussian')
            results['tv_weights_effective'] = manifest['tv_weights_effective']
            if args.regularization == 'tv':
                shape, spacing = ((nx, nz), (dX, dZ)) if escalate else ((nz,), (dZ,))
                cells = int(np.prod(shape))
                results['posterior_tv'] = [tv.value_and_gradient(np.asarray(param)[:cells], shape,
                                                                spacing, args.tv_epsilon)[0]
                                            for param in results['posterior_params']]
            results['misfit']['after_final_kind'] = kind
            if linearized:
                results['prior_forward_predictions'] = [record['initial_prediction'] for record in records]
                results['prior_forward_misfit'] = [measurement_misfit(record['initial_prediction'],
                                                                      observations, Cd) for record in records]
            with open(f'inversion_results_assim_{index}.pkl', 'xb') as stream:
                pickle.dump(results, stream)
            previous_index = index
    finally:
        os.chdir(previous_cwd)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
