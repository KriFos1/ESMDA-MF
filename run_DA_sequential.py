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
from scipy.stats import norm

from esmda_core import (LOG_RH_LIMITS, LocalWindow, MemberEvaluationError, clamp_log_rh,
                        measurement_misfit, noise_floor_mask, observation_vectors,
                        optimizer_jacobian, optimize_members, project_to_modes,
                        update_members_once)
from esmda_remote2d import UDAR_COMPONENTS, SimBackendConfig, make_remote2d
import tv_regularization as tv

meter_to_feet = 3.28084
dims = (15, 5, 20)
dX = dY = dZ = 5.0
min_ratio, max_ratio = 1.0, 5.0
tools = ["('6kHz','83ft')", "('12kHz','83ft')", "('24kHz','83ft')",
         "('24kHz','43ft')", "('48kHz','43ft')", "('96kHz','43ft')"]
masked_udar_components = ('USDP', 'USDA')
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


def conditional_columns(base_rh, base_latent, background, reservoir, mode_std, latent_std, switch_prob):
    """Blocky conditional draw of fresh cells from the adjacent ensemble column.

    Modes persist laterally with per-cell flips at ``switch_prob``; values are
    redrawn at the mode level with ``mode_std`` spread. Formula-based: no
    iterative solves, so entering cells cannot stall.
    """
    midpoint = 0.5 * (background + reservoir)
    flips = np.random.random(base_rh.shape) < switch_prob
    reservoir_cells = base_rh > midpoint
    rh = base_rh + np.random.normal(0., mode_std, base_rh.shape)
    rh = np.where(flips & reservoir_cells,
                  np.random.normal(background, mode_std, base_rh.shape), rh)
    rh = np.where(flips & ~reservoir_cells,
                  np.random.normal(reservoir, mode_std, base_rh.shape), rh)
    latent = base_latent + np.random.normal(0., latent_std, base_latent.shape)
    return rh, latent


def conditional_row(base_rh, base_latent, background, reservoir, mode_std, latent_std, switch_prob):
    """One new depth row per column, conditioned on the deepest existing row."""
    midpoint = 0.5 * (background + reservoir)
    flips = np.random.random(base_rh.shape) < switch_prob
    reservoir_cells = base_rh > midpoint
    reservoir_row = np.logical_xor(flips, reservoir_cells)
    rh = np.where(reservoir_row,
                  np.random.normal(reservoir, mode_std, base_rh.shape),
                  np.random.normal(background, mode_std, base_rh.shape))
    latent = base_latent + np.random.normal(0., latent_std, base_latent.shape)
    return rh, latent


def sample_prior_2d(ne, zero_d_posterior, background, reservoir, mode_std, fraction,
                    switch_prob, projection=True):
    """Laterally persistent two-mode blocky window prior.

    The bit column follows ``sample_blocky_prior_1D`` statistics conditioned on
    the member's 0D posterior side; interfaces random-walk laterally through
    ``conditional_columns`` so the ensemble carries real cross-column
    correlation for the ES updates while new geology stays discoverable.
    """
    nx, _, nz = dims
    columns, _, _ = sample_blocky_prior_1D(ne, zero_d_posterior, background, reservoir,
                                           mode_std, fraction, projection=projection)
    rh = np.empty((nx, nz, ne))
    latent = np.empty((nx, nz, ne))
    bit = nx // 2
    rh[bit], latent[bit] = columns[:nz], columns[nz:]
    for column in range(bit - 1, -1, -1):
        rh[column], latent[column] = conditional_columns(
            rh[column + 1], latent[column + 1], background, reservoir,
            mode_std, ratio_latent_std, switch_prob)
    for column in range(bit + 1, nx):
        rh[column], latent[column] = conditional_columns(
            rh[column - 1], latent[column - 1], background, reservoir,
            mode_std, ratio_latent_std, switch_prob)
    if projection:
        for column in range(nx):
            rh[column] = project_to_modes(rh[column], background, reservoir, 2. * mode_std)
    state = np.vstack([rh.reshape(nx * nz, ne), latent.reshape(nx * nz, ne)])
    mean = state.mean(axis=1)
    covariance = np.diag(np.maximum(state.var(axis=1), 1e-8))
    return state, covariance, mean


def shift_window_state(state, movement, background, reservoir, mode_std, latent_std, switch_prob):
    """Reindex the moving window; entered cells become conditional blocky draws.

    Exited cells are dropped and finalized. Cells entering laterally (or at the
    bottom on vertical movement) are drawn from the adjacent interior ensemble
    column/row, preserving correlation with the informed state.
    """
    nx, _, nz = dims
    values = np.asarray(state, dtype=float)
    if values.ndim != 2 or values.shape[0] != 2 * nx * nz:
        raise ValueError('Window state must have shape (2*nx*nz, members)')
    ne = values.shape[1]
    x = np.clip(np.arange(nx) + movement[0], 0, nx - 1)
    z = np.clip(np.arange(nz) + movement[1], 0, nz - 1)
    cells = nx * nz
    rh = values[:cells].reshape(nx, nz, ne)[x][:, z].copy()
    latent = values[cells:].reshape(nx, nz, ne)[x][:, z].copy()
    entered_x = int(movement[0])
    if entered_x > 0:
        for column in range(nx - entered_x, nx):
            rh[column], latent[column] = conditional_columns(
                rh[column - 1], latent[column - 1], background, reservoir,
                mode_std, latent_std, switch_prob)
    elif entered_x < 0:
        for column in range(-entered_x - 1, -1, -1):
            rh[column], latent[column] = conditional_columns(
                rh[column + 1], latent[column + 1], background, reservoir,
                mode_std, latent_std, switch_prob)
    entered_z = int(movement[1])
    if entered_z > 0:
        for row in range(nz - entered_z, nz):
            rh[:, row], latent[:, row] = conditional_row(
                rh[:, row - 1], latent[:, row - 1], background, reservoir,
                mode_std, latent_std, switch_prob)
    elif entered_z < 0:
        for row in range(-entered_z - 1, -1, -1):
            rh[:, row], latent[:, row] = conditional_row(
                rh[:, row + 1], latent[:, row + 1], background, reservoir,
                mode_std, latent_std, switch_prob)
    return np.vstack([rh.reshape(cells, ne), latent.reshape(cells, ne)])


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
    parser.add_argument('--prior-style', choices=['blocky'], default='blocky',
                        help='Unified-domain two-mode blocky prior')
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
                     'unified_domain': True,
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
        previous_index = None
        for index in range(args.start, stop, args.frequency):
            point_started = time.monotonic()
            current_shift, movement = window.update((WX[index], WY[index], TVD[index]))
            _set_logging_point_state(index)
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
            nx, _, nz = dims
            cells = nx * nz
            bit = nx // 2
            ne = args.ensemble_size
            threshold = args.misfit_threshold
            misfit = {}
            tv_history = []
            escalation_abandoned = None
            solver_failures = []
            alpha, used_weight = [10.], [0.]
            spatial_weight = [None]
            updater = approx_update()
            updater.proj = (np.eye(ne) - np.ones((ne, ne)) / ne) / np.sqrt(ne - 1)
            updater.trunc_energy, updater.keys_da, updater.lam = .95, {}, 0
            updater.list_states = ['rh', 'rv']

            def predict_state(state, fidelity):
                if fidelity == '0D' and state.shape[0] == 2:
                    columns = [state[:, member] for member in range(ne)]
                elif fidelity == '0D':
                    columns = [np.array([state[:cells, member].mean(),
                                         state[cells:, member].mean()]) for member in range(ne)]
                else:
                    upper, lower = bit * nz, cells + bit * nz
                    columns = [np.r_[state[upper:upper + nz, member],
                                     state[lower:lower + nz, member]] for member in range(ne)]
                return np.column_stack([simulate_pred_only(column, index, fidelity)
                                        for column in columns])[obs_mask]

            def assimilate_stage(state, fidelity, budget):
                updater.state_scaling = np.ones(state.shape[0])
                active = state.shape[0] // 2
                updater.prior_info = {key: {'active': active} for key in ('rh', 'rv')}
                prediction = predict_state(state, fidelity)
                misfit_now = measurement_misfit(prediction.mean(axis=1), observations, Cd)
                misfit[f'{fidelity}_initial'] = misfit_now
                while misfit_now > threshold and used_weight[0] < budget:
                    inflated = Cd * alpha[0]
                    updater.scale_data = np.sqrt(inflated)
                    perturbed = np.random.normal(observations[:, None], np.sqrt(inflated)[:, None],
                                                 (len(Cd), ne))
                    updater.update(enX=state, enY=prediction, enE=perturbed)
                    state += updater.step
                    np.clip(state[:active], *LOG_RH_LIMITS, out=state[:active])
                    if projection:
                        state[:active] = project_to_modes(state[:active], background, reservoir,
                                                          2. * mode_std)
                    if args.regularization == 'tv' and fidelity == '1D':
                        dose = tv_weight_1d / alpha[0] / spatial_weight[0]
                        state = tv.regularize_windows(state, nx, nz, dose, dX, dZ,
                                                      args.tv_epsilon, args.tv_iterations)
                        tv_history.append({'stage': '1D', 'strength': dose,
                                           'assimilation_fraction': 1 / alpha[0] / spatial_weight[0]})
                    used_weight[0] += 1 / alpha[0]
                    alpha[0] *= 10 / 9
                    prediction = predict_state(state, fidelity)
                    misfit_now = measurement_misfit(prediction.mean(axis=1), observations, Cd)
                    print(f'Index {index}, {fidelity}: misfit={misfit_now:.6g}, '
                          f'weight={used_weight[0]:.4f}', flush=True)
                misfit[f'{fidelity}_final'] = misfit_now
                return state, prediction

            if previous is None:
                state = clamp_log_rh(sample_prior_0D(ne, background, reservoir, mode_std,
                                                     args.prior_reservoir_fraction)[0])
                state, _ = assimilate_stage(state, '0D', .25)
                state = clamp_log_rh(sample_prior_2d(ne, state[0], background, reservoir, mode_std,
                                                     args.prior_reservoir_fraction,
                                                     args.prior_switch_prob, projection=projection)[0])
                point_prior = state.copy()
                spatial_weight[0] = 1. - used_weight[0]
                state, prediction = assimilate_stage(state, '1D', .75)
            else:
                state = clamp_log_rh(shift_window_state(previous, movement, background, reservoir,
                                                        mode_std, ratio_latent_std,
                                                        args.prior_switch_prob))
                point_prior = state.copy()
                state, _ = assimilate_stage(state, '0D', .25)
                spatial_weight[0] = 1. - used_weight[0]
                state, prediction = assimilate_stage(state, '1D', .75)
            post_1d_state = state.copy()
            mean = state.mean(axis=1)
            covariance = np.diag(np.maximum(state.var(axis=1), 1e-8))
            ranking_full = np.argsort([custom_loss(prediction[:, member], observations,
                                                   state[:, member], mean, covariance, Cd)
                                       for member in range(ne)])
            ranking = ranking_full[:args.opt_members]
            factor = 1 / (1 - used_weight[0])
            inflated = Cd * factor
            final_tv_fraction = 1 / factor / spatial_weight[0]
            final_tv_dose = tv_weight_2d * final_tv_fraction
            escalate = args.force_2d or misfit['1D_final'] > threshold
            mean_2d, covariance_2d = mean, covariance
            two_d_evaluations = [0]
            tv_history_mark = len(tv_history)
            records = None
            member_candidates = [int(rank) for rank in ranking]
            posterior_misfit = None
            worsened = False
            if escalate:
                prior = post_1d_state[:, ranking].copy()
                mean_2d = prior.mean(axis=1)
                covariance_2d = np.diag(np.maximum(prior.var(axis=1), 1e-8))
                perturbed = np.random.normal(observations[:, None], np.sqrt(inflated)[:, None],
                                             (len(Cd), args.opt_members))
                active_cells = np.ones((2, nx, nz), dtype=bool)
                active_cells[:, :bit] = False

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
                    prior[:, slot] = post_1d_state[:, candidate].copy()
                    perturbed[:, slot] = np.random.normal(observations, np.sqrt(inflated))
                    member_candidates[slot] = candidate
                    return prior[:, slot], perturbed[:, slot]

                try:
                    if args.single_update:
                        update = map_update_data_space_LM
                        if args.regularization == 'tv':
                            def update(theta, member_prediction, prior_mean, jacobian, datum,
                                       data_cov, prior_cov, damping):
                                return tv.linearized_step(
                                    theta, member_prediction, prior_mean, jacobian, datum, data_cov,
                                    (nx, nz), (dX, dZ), final_tv_dose, active_cells.ravel(),
                                    args.tv_epsilon, args.tv_iterations, args.tv_amplitude_std,
                                    args.tv_max_step, LOG_RH_LIMITS)
                            tv_history.append({'stage': '2D', 'strength': final_tv_dose,
                                               'assimilation_fraction': 1 / factor / spatial_weight[0]})
                        records = update_members_once([prior[:, slot] for slot in range(args.opt_members)],
                                                       [perturbed[:, slot] for slot in range(args.opt_members)],
                                                       mean_2d, covariance_2d, inflated, evaluate_multi_2d,
                                                       update, custom_loss,
                                                       active_cells.ravel(), substitute=substitute)
                    else:
                        records = optimize_members([prior[:, slot] for slot in range(args.opt_members)],
                                                   [perturbed[:, slot] for slot in range(args.opt_members)],
                                                   mean_2d, covariance_2d, inflated, evaluate_multi_2d,
                                                   map_update_data_space_LM, custom_loss,
                                                   args.max_iterations, args.max_lm_trials,
                                                   active_cells.ravel(), substitute=substitute)
                except MemberEvaluationError as exc:
                    escalation_abandoned = str(exc)
                    print(f'Index {index}: 2D escalation abandoned after '
                          f'{len(solver_failures)} substitutions; applying final 1D update', flush=True)
                    del tv_history[tv_history_mark:]
                    escalate = False
                else:
                    for slot, record in enumerate(records):
                        label = 'gaussian_diagnostic_loss' if args.regularization == 'tv' else 'loss'
                        print(f'Index {index}, 2D member {slot+1} (candidate {member_candidates[slot]}): '
                              f'{label}={record["loss"]:.6g}, iterations={len(record["history"]) - 1}',
                              flush=True)
                    misfit['before_final'] = misfit['1D_final']
                    posterior_misfit = [measurement_misfit(record['prediction'], observations, Cd)
                                        for record in records]
                    misfit['after_final'] = float(np.mean(posterior_misfit))
            if records is None:
                misfit['before_final'] = misfit['1D_final']
                updater.state_scaling = np.ones(state.shape[0])
                active = state.shape[0] // 2
                updater.prior_info = {key: {'active': active} for key in ('rh', 'rv')}
                updater.scale_data = np.sqrt(inflated)
                perturbed = np.random.normal(observations[:, None], np.sqrt(inflated)[:, None],
                                             (len(Cd), ne))
                updater.update(enX=state, enY=prediction, enE=perturbed)
                state += updater.step
                np.clip(state[:active], *LOG_RH_LIMITS, out=state[:active])
                if projection:
                    state[:active] = project_to_modes(state[:active], background, reservoir,
                                                      2. * mode_std)
                if args.regularization == 'tv':
                    final_dose = tv_weight_1d * final_tv_fraction
                    state = tv.regularize_windows(state, nx, nz, final_dose, dX, dZ,
                                                  args.tv_epsilon, args.tv_iterations)
                    tv_history.append({'stage': 'final_1D', 'strength': final_dose,
                                       'assimilation_fraction': final_tv_fraction})
                used_weight[0] += 1 / factor
                prediction = predict_state(state, '1D')
                misfit_now = measurement_misfit(prediction.mean(axis=1), observations, Cd)
                misfit['after_final'] = misfit_now
                worsened = bool(misfit_now > misfit['before_final'])
                print(f'Index {index}, final 1D update: misfit {misfit["before_final"]:.6g} -> '
                      f'{misfit_now:.6g}{" (increased)" if worsened else ""}, weight=1.0000', flush=True)
                ranking = np.argsort([custom_loss(prediction[:, member], perturbed[:, member],
                                                  state[:, member], mean, covariance, inflated)
                                      for member in range(ne)])[:args.opt_members]
                member_candidates = [int(rank) for rank in ranking]
                records = [{'parameters': state[:, ranking[member]].copy(),
                            'prediction': prediction[:, ranking[member]],
                            'loss': float(custom_loss(prediction[:, ranking[member]],
                                                      perturbed[:, member],
                                                      state[:, ranking[member]], mean,
                                                      covariance, inflated)),
                            'history': [], 'trial_errors': []}
                           for member in range(args.opt_members)]
                posterior_misfit = [measurement_misfit(record['prediction'], observations, Cd)
                                    for record in records]
                for record in records:
                    record['history'] = [record['loss']]
            engaged = escalate and records is not None
            if engaged:
                state = state.copy()
                for slot, record in enumerate(records):
                    state[:, member_candidates[slot]] = record['parameters']
            linearized = engaged and args.single_update
            kind = 'linearized' if linearized else 'forward'
            selected_fidelity = '2D' if engaged else '1D'
            two_d_mode = ('single-update' if args.single_update else 'lm') if engaged \
                else ('abandoned' if escalation_abandoned is not None else None)
            results = {'prior_params': point_prior, 'post_mda_params': post_1d_state.copy(),
                       'post_1d_params': post_1d_state.copy(),
                       'posterior_params': [record['parameters'] for record in records],
                       'posterior_predictions': [record['prediction'] for record in records],
                       'posterior_losses': [record['loss'] for record in records],
                       'optimizer_history': [record['history'] for record in records],
                       'trial_errors': [record['trial_errors'] for record in records],
                       'observations': observations, 'perturbed_observations': perturbed,
                       'data_covariance': inflated,
                       'prior_mean': mean_2d if engaged else mean,
                       'prior_covariance': covariance_2d if engaged else covariance,
                       'dims': dims, 'spacing_ft': (dX, dY, dZ), 'shift_ft': dict(current_shift),
                       'window_movement': movement, 'previous_index': previous_index,
                       'logging_index': index, 'tools': tool_pairs, 'seed': args.seed,
                       'mda_weight': used_weight[0],
                       'lower_fidelity_weight': used_weight[0] - (0. if engaged else 1. / factor),
                       'final_inflation_factor': factor,
                       'optimized_2d': engaged, 'selected_fidelity': selected_fidelity,
                       'two_d_calls': two_d_evaluations[0], 'two_d_mode': two_d_mode,
                       'prior_style': args.prior_style,
                       'misfit': misfit, 'misfit_threshold': threshold,
                       'posterior_misfit': posterior_misfit,
                       'final_iteration_worsened': worsened,
                       'member_candidates': member_candidates,
                       'solver_failures': solver_failures,
                       'substituted_members': len(solver_failures),
                       'observation_mask': obs_mask, 'masked_channels': masked_channels,
                       'masked_components': masked_udar_components}
            if escalation_abandoned is not None:
                results['escalation_abandoned'] = escalation_abandoned
            results.update(unified_domain=True, parameter_units='natural_log',
                           prior_projection=args.prior_projection,
                           prior_units=manifest['prior_units'],
                           posterior_prediction_kind=kind, posterior_misfit_kind=kind,
                           posterior_forward_verified=not linearized,
                           point_elapsed_seconds=time.monotonic() - point_started)
            results.update(regularization=args.regularization, tv_weight=args.tv_weight,
                           tv_epsilon=args.tv_epsilon, tv_iterations=args.tv_iterations,
                           tv_history=tv_history, prior_projection_effective=projection,
                           tv_strength_applied=sum(entry['strength'] for entry in tv_history),
                           loss_kind='gaussian_diagnostic' if args.regularization == 'tv'
                           else 'gaussian')
            results['tv_weights_effective'] = manifest['tv_weights_effective']
            if args.regularization == 'tv':
                results['posterior_tv'] = [tv.value_and_gradient(np.asarray(param)[:cells],
                                                                 (nx, nz), (dX, dZ),
                                                                 args.tv_epsilon)[0]
                                           for param in results['posterior_params']]
            results['misfit']['after_final_kind'] = kind
            if linearized:
                results['prior_forward_predictions'] = [record['initial_prediction']
                                                        for record in records]
                results['prior_forward_misfit'] = [measurement_misfit(record['initial_prediction'],
                                                                      observations, Cd)
                                                   for record in records]
            with open(f'inversion_results_assim_{index}.pkl', 'xb') as stream:
                pickle.dump(results, stream)
            previous = state
            previous_index = index
    finally:
        os.chdir(previous_cwd)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
