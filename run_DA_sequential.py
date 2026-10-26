"""Sequential 0D/1D ES-MDA and local-window 2D gradient inversion."""

import argparse
import ast
from dataclasses import asdict
import hashlib
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
                        log_rv_from_latent, optimizer_jacobian, optimize_members,
                        project_to_modes, update_members_once)
from esmda_remote2d import (UDAR_COMPONENTS, SimBackendConfig,
                            classify_solver_start_failure, make_remote2d)
import tv_regularization as tv

meter_to_feet = 3.28084
dims = (15, 5, 20)
dX = dY = dZ = 5.0
min_ratio, max_ratio = 1.0, 3.0
# Anisotropy state -> ln(Rv/Rh): 'probit' (bounded ratio) or 'log_clipped' (max(latent, 0)).
anisotropy = 'probit'
rv_cap_ohmm = 80.0
log_rv_cap = float(np.log(rv_cap_ohmm))
tools = ["('6kHz','83ft')", "('12kHz','83ft')", "('24kHz','83ft')",
         "('24kHz','43ft')", "('48kHz','43ft')", "('96kHz','43ft')"]
masked_udar_components = ('USDP', 'USDA')
ratio_latent_mean = 0.0
ratio_latent_std = 0.12


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
              'map': {'ratio': [min_ratio, max_ratio]}, 'obsname': 'tvd',
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


def conditional_mixture_columns(base_rh, base_latent, background, reservoir, mode_std,
                                latent_std, min_block_cells, interface_jitter,
                                value_correlation, within_block_std):
    """Extend a correlated block mixture by jittering whole interfaces only."""
    nz, members = base_rh.shape
    midpoint = .5 * (background + reservoir)
    rh, latent = np.empty_like(base_rh), np.empty_like(base_latent)
    for member in range(members):
        base_modes = base_rh[:, member] > midpoint
        old_bounds = np.r_[0, np.flatnonzero(np.diff(base_modes.astype(int)) != 0) + 1, nz]
        bounds = old_bounds.copy()
        lengths = np.diff(old_bounds)
        minimum = max(1, min(int(min_block_cells), int(lengths.min())))
        for j in range(1, len(bounds) - 1):
            candidate = int(old_bounds[j])
            if np.random.random() < interface_jitter:
                candidate += int(np.random.choice((-1, 1)))
            lower = bounds[j - 1] + minimum
            remaining_segments = len(bounds) - 1 - j
            upper = nz - remaining_segments * minimum
            bounds[j] = int(np.clip(candidate, lower, upper))
        mode_at_top = bool(base_modes[0])
        for segment, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            is_reservoir = mode_at_top if segment % 2 == 0 else not mode_at_top
            mode_mean = reservoir if is_reservoir else background
            old_lo, old_hi = old_bounds[segment], old_bounds[segment + 1]
            old_level = float(base_rh[old_lo:old_hi, member].mean())
            level = mode_mean + value_correlation * (old_level - mode_mean) \
                + np.random.normal(0., mode_std * np.sqrt(1. - value_correlation ** 2))
            rh[lo:hi, member] = level + _ar1_noise(
                hi - lo, 1, .90, within_block_std)[:, 0]
            old_latent_level = float(base_latent[old_lo:old_hi, member].mean())
            latent_level = value_correlation * old_latent_level \
                + np.random.normal(0., latent_std * np.sqrt(1. - value_correlation ** 2))
            latent[lo:hi, member] = latent_level + _ar1_noise(
                hi - lo, 1, .90, min(.05, latent_std * .2))[:, 0]
    return rh, latent


def conditional_gaussian_columns(base_rh, base_latent, mean_rh, std_rh, latent_std,
                                  rho_x, rho_z, rho_lat_z=None, rho_lat_x=None):
    if rho_lat_z is None:
        rho_lat_z = rho_z
    if rho_lat_x is None:
        rho_lat_x = rho_x
    innovation = np.sqrt(1. - rho_x ** 2)
    latent_innovation = np.sqrt(1. - rho_lat_x ** 2)
    rh_noise = _ar1_noise(base_rh.shape[0], base_rh.shape[1], rho_z)
    latent_noise = _ar1_noise(base_latent.shape[0], base_latent.shape[1], rho_lat_z)
    rh = mean_rh + rho_x * (base_rh - mean_rh) + std_rh * innovation * rh_noise
    latent = rho_lat_x * base_latent + latent_std * latent_innovation * latent_noise
    return rh, latent


def conditional_gaussian_row(base_rh, base_latent, mean_rh, std_rh, latent_std,
                              rho_x, rho_z, rho_lat_z=None, rho_lat_x=None):
    if rho_lat_z is None:
        rho_lat_z = rho_z
    if rho_lat_x is None:
        rho_lat_x = rho_x
    innovation = np.sqrt(1. - rho_z ** 2)
    latent_innovation = np.sqrt(1. - rho_lat_z ** 2)
    rh_noise = _ar1_noise(base_rh.shape[0], base_rh.shape[1], rho_x)
    latent_noise = _ar1_noise(base_latent.shape[0], base_latent.shape[1], rho_lat_x)
    rh = mean_rh + rho_z * (base_rh - mean_rh) + std_rh * innovation * rh_noise
    latent = rho_lat_z * base_latent + latent_std * latent_innovation * latent_noise
    return rh, latent


def conditional_mixture_row(base_rh, base_latent, background, reservoir, mode_std,
                            latent_std, value_correlation, within_block_std):
    """Extend a block-mixture bottom/top edge without independent mode flips."""
    midpoint = .5 * (background + reservoir)
    modes = base_rh > midpoint
    mode_mean = np.where(modes, reservoir, background)
    rh = mode_mean + value_correlation * (base_rh - mode_mean) \
        + np.random.normal(0., mode_std * np.sqrt(1. - value_correlation ** 2),
                           base_rh.shape)
    rh += _ar1_noise(base_rh.shape[0], base_rh.shape[1], .90, within_block_std)
    latent = value_correlation * base_latent \
        + np.random.normal(0., latent_std * np.sqrt(1. - value_correlation ** 2),
                           base_latent.shape)
    return rh, latent


def _ar1_noise(length, members, rho, std=1.):
    values = np.empty((length, members))
    values[0] = np.random.normal(0., std, members)
    innovation = np.sqrt(max(1. - rho ** 2, 0.)) * std
    for index in range(1, length):
        values[index] = rho * values[index - 1] + np.random.normal(0., innovation, members)
    return values


def _ar1_field(nx, nz, members, rho_x, rho_z):
    values = np.random.normal(size=(nx, nz, members))
    sx, sz = np.sqrt(max(1. - rho_x ** 2, 0.)), np.sqrt(max(1. - rho_z ** 2, 0.))
    for z in range(1, nz):
        values[:, z, :] = rho_z * values[:, z - 1, :] + sz * values[:, z, :]
    for x in range(1, nx):
        values[x, :, :] = rho_x * values[x - 1, :, :] + sx * values[x, :, :]
    return values


def _block_lengths(nz, switches, min_cells):
    min_cells = min(max(1, int(min_cells)), nz)
    segments = min(int(switches) + 1, max(1, nz // min_cells))
    switches = segments - 1
    remaining = nz - segments * min_cells
    additions = np.random.multinomial(remaining, np.full(segments, 1. / segments))
    lengths = min_cells + additions
    boundaries = np.cumsum(lengths[:-1]).astype(int)
    return boundaries


def sample_long_block_prior_2d(ne, zero_d_posterior, background, reservoir, mode_std,
                               fraction, nx, nz, min_block_cells=10,
                               fault_fraction=.15, fault_min_block_cells=3,
                               within_block_std=.046, interface_jitter=.01,
                               value_correlation=.98, latent_std=.25):
    """Two-mode fields with minimum vertical blocks and coherent lateral interfaces."""
    midpoint = .5 * (background + reservoir)
    bit = nx // 2
    rh = np.empty((nx, nz, ne))
    latent = np.empty((nx, nz, ne))
    fault_members = np.random.random(ne) < fault_fraction
    bit_modes = np.empty((nz, ne), dtype=bool)
    bit_bounds = []
    for member in range(ne):
        minimum = max(1, int(fault_min_block_cells if fault_members[member] else min_block_cells))
        max_switches = max(0, nz // minimum - 1)
        if fault_members[member]:
            switch_count = np.random.randint(1, min(6, max_switches) + 1) if max_switches else 0
        else:
            allowed = np.array([k for k in (0, 1, 2, 3) if k <= max_switches], dtype=int)
            probability_by_count = {0: .20, 1: .25, 2: .40, 3: .15}
            probs = np.array([probability_by_count[k] for k in allowed], dtype=float)
            probs /= probs.sum()
            switch_count = int(np.random.choice(allowed, p=probs))
        bounds = np.r_[0, _block_lengths(nz, switch_count, minimum), nz].astype(int)
        center_segment = int(np.searchsorted(bounds, nz // 2, side='right')) - 1
        center_is_reservoir = bool(zero_d_posterior[member] > midpoint)
        start_is_reservoir = np.random.random() < fraction
        states = [start_is_reservoir if i % 2 == 0 else not start_is_reservoir
                  for i in range(len(bounds) - 1)]
        states[center_segment] = center_is_reservoir
        merged_bounds = [int(bounds[0])]
        merged_states = [states[0]]
        for segment, boundary in enumerate(bounds[1:-1], start=1):
            if states[segment] != states[segment - 1]:
                merged_bounds.append(int(boundary))
                merged_states.append(states[segment])
        merged_bounds.append(nz)
        bounds, states = np.asarray(merged_bounds, dtype=int), merged_states
        modes = np.empty(nz, dtype=bool)
        latent_column = np.empty(nz)
        rh_column = np.empty(nz)
        for segment, (lo, hi, is_reservoir) in enumerate(zip(bounds[:-1], bounds[1:], states)):
            mode_mean = reservoir if is_reservoir else background
            level = np.random.normal(mode_mean, mode_std)
            rh_column[lo:hi] = level + _ar1_noise(hi - lo, 1, .90, within_block_std)[:, 0]
            latent_level = np.random.normal(0., latent_std)
            latent_column[lo:hi] = latent_level + _ar1_noise(
                hi - lo, 1, .90, min(.05, latent_std * .2))[:, 0]
            modes[lo:hi] = is_reservoir
        rh[bit, :, member] = rh_column
        latent[bit, :, member] = latent_column
        bit_modes[:, member] = modes
        bit_bounds.append(bounds)

    for direction in (-1, 1):
        columns = range(bit - 1, -1, -1) if direction < 0 else range(bit + 1, nx)
        previous = bit
        previous_bounds = bit_bounds
        previous_modes = [bit_modes[:, member].copy() for member in range(ne)]
        for column in columns:
            next_bounds = []
            next_modes = []
            for member in range(ne):
                old_bounds = previous_bounds[member]
                old_rh = rh[previous, :, member]
                old_latent_column = latent[previous, :, member]
                bounds = old_bounds.copy()
                minimum = max(1, int(fault_min_block_cells if fault_members[member]
                                      else min_block_cells))
                for j in range(1, len(bounds) - 1):
                    if np.random.random() < interface_jitter:
                        bounds[j] += int(np.random.choice((-1, 1)))
                    lower = bounds[j - 1] + minimum
                    upper = bounds[j + 1] - minimum
                    bounds[j] = int(np.clip(bounds[j], lower, upper))
                modes = np.empty(nz, dtype=bool)
                rh_column = np.empty(nz)
                latent_column = np.empty(nz)
                first_mode = previous_modes[member][0]
                for j, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
                    is_reservoir = bool(first_mode if j % 2 == 0 else not first_mode)
                    mode_mean = reservoir if is_reservoir else background
                    old_lo, old_hi = old_bounds[j], old_bounds[j + 1]
                    old_level = float(old_rh[old_lo:old_hi].mean())
                    level = mode_mean + value_correlation * (old_level - mode_mean) \
                        + np.random.normal(0., mode_std * np.sqrt(1. - value_correlation ** 2))
                    rh_column[lo:hi] = level + _ar1_noise(
                        hi - lo, 1, .90, within_block_std)[:, 0]
                    old_latent_level = float(old_latent_column[old_lo:old_hi].mean())
                    latent_level = value_correlation * old_latent_level \
                        + np.random.normal(0., latent_std * np.sqrt(1. - value_correlation ** 2))
                    latent_column[lo:hi] = latent_level + _ar1_noise(
                        hi - lo, 1, .90, min(.05, latent_std * .2))[:, 0]
                    modes[lo:hi] = is_reservoir
                rh[column, :, member] = rh_column
                latent[column, :, member] = latent_column
                next_bounds.append(bounds)
                next_modes.append(modes)
            previous_bounds = next_bounds
            previous_modes = next_modes
            previous = column
    return np.vstack([rh.reshape(nx * nz, ne), latent.reshape(nx * nz, ne)])


def sample_gaussian_prior_2d(ne, zero_d_posterior, background, reservoir, mode_std,
                             fraction, nx, nz, dx, dz, corr_x_ft=80., corr_z_ft=75.,
                             latent_corr_z_ft=35., latent_std=.12,
                             latent_corr_x_ft=80.):
    """Draw a smooth Gaussian field, optionally matching a supplied bit-center state."""
    mean_rh = fraction * reservoir + (1. - fraction) * background
    variance_rh = mode_std ** 2 + fraction * (1. - fraction) * (reservoir - background) ** 2
    std_rh = np.sqrt(variance_rh)
    rho_x, rho_z = np.exp(-dx / corr_x_ft), np.exp(-dz / corr_z_ft)
    rho_lat_x = np.exp(-dx / latent_corr_x_ft)
    rho_lat_z = np.exp(-dz / latent_corr_z_ft)
    rh = mean_rh + std_rh * _ar1_field(nx, nz, ne, rho_x, rho_z)
    latent = latent_std * _ar1_field(nx, nz, ne, rho_lat_x, rho_lat_z)
    if zero_d_posterior is not None:
        bit, center_z = nx // 2, nz // 2
        cx = np.exp(-abs(np.arange(nx) - bit)[:, None] * dx / corr_x_ft)
        cz = np.exp(-abs(np.arange(nz) - center_z)[None, :] * dz / corr_z_ft)
        weights = cx * cz
        for member in range(ne):
            rh[:, :, member] += weights * (zero_d_posterior[member] - rh[bit, center_z, member])
    state = np.vstack([rh.reshape(nx * nz, ne), latent.reshape(nx * nz, ne)])
    return state


def sample_prior_2d(ne, zero_d_posterior, background, reservoir, mode_std, fraction,
                    switch_prob, projection=True, prior_style='blocky', dx=5., dz=5.,
                    min_block_cells=10, fault_fraction=.15, fault_min_block_cells=3,
                    within_block_std=.046, interface_jitter=.01, value_correlation=.98,
                    corr_x_ft=10., corr_z_ft=75., latent_corr_z_ft=35.,
                    latent_corr_x_ft=80.):
    """Laterally persistent two-mode blocky window prior.

    The bit column follows ``sample_blocky_prior_1D`` statistics conditioned on
    the member's 0D posterior side; interfaces random-walk laterally through
    ``conditional_columns`` so the ensemble carries real cross-column
    correlation for the ES updates while new geology stays discoverable.
    """
    nx, _, nz = dims
    if prior_style == 'blocky':
        columns, _, _ = sample_blocky_prior_1D(ne, zero_d_posterior, background, reservoir,
                                               mode_std, fraction, projection=projection)
        rh = np.empty((nx, nz, ne)); latent = np.empty((nx, nz, ne))
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
    elif prior_style == 'gaussian':
        state = sample_gaussian_prior_2d(ne, zero_d_posterior, background, reservoir,
                                         mode_std, fraction, nx, nz, dx, dz,
                                         corr_x_ft, corr_z_ft, latent_corr_z_ft,
                                         ratio_latent_std, latent_corr_x_ft)
    elif prior_style == 'mixture':
        state = sample_long_block_prior_2d(ne, zero_d_posterior, background, reservoir,
                                           mode_std, fraction, nx, nz, min_block_cells,
                                           fault_fraction, fault_min_block_cells,
                                           within_block_std, interface_jitter,
                                           value_correlation, ratio_latent_std)
    else:
        raise ValueError(f'Unknown prior family {prior_style}')
    mean = state.mean(axis=1)
    covariance = np.maximum(state.var(axis=1), 1e-8)
    return state, covariance, mean


def shift_window_state(state, movement, background, reservoir, mode_std, latent_std, switch_prob,
                      prior_style='blocky', fraction=.15, dx=5., dz=5.,
                      min_block_cells=10, fault_fraction=.15, fault_min_block_cells=3,
                      within_block_std=.046, interface_jitter=.01, value_correlation=.98,
                      corr_x_ft=10., corr_z_ft=75., latent_corr_z_ft=35.,
                      latent_corr_x_ft=80.):
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
    mixture_mean = fraction * reservoir + (1. - fraction) * background
    mixture_var = mode_std ** 2 + fraction * (1. - fraction) * (reservoir - background) ** 2
    mixture_std = np.sqrt(mixture_var)

    rho_lat_z = np.exp(-dz / latent_corr_z_ft)
    rho_lat_x = np.exp(-dx / latent_corr_x_ft)

    def extend_column(base_rh, base_latent):
        if prior_style == 'blocky':
            return conditional_columns(base_rh, base_latent, background, reservoir,
                                       mode_std, latent_std, switch_prob)
        if prior_style == 'mixture':
            return conditional_mixture_columns(base_rh, base_latent, background, reservoir,
                                               mode_std, latent_std, min_block_cells,
                                               interface_jitter, value_correlation,
                                               within_block_std)
        return conditional_gaussian_columns(
            base_rh, base_latent, mixture_mean, mixture_std, latent_std,
            np.exp(-dx / corr_x_ft), np.exp(-dz / corr_z_ft), rho_lat_z, rho_lat_x)

    def extend_row(base_rh, base_latent):
        if prior_style == 'blocky':
            return conditional_row(base_rh, base_latent, background, reservoir,
                                   mode_std, latent_std, switch_prob)
        if prior_style == 'mixture':
            return conditional_mixture_row(base_rh, base_latent, background, reservoir,
                                            mode_std, latent_std, value_correlation,
                                            within_block_std)
        return conditional_gaussian_row(
            base_rh, base_latent, mixture_mean, mixture_std, latent_std,
            np.exp(-dx / corr_x_ft), np.exp(-dz / corr_z_ft), rho_lat_z, rho_lat_x)

    entered_x = int(movement[0])
    if entered_x > 0:
        for column in range(nx - entered_x, nx):
            rh[column], latent[column] = extend_column(rh[column - 1], latent[column - 1])
    elif entered_x < 0:
        for column in range(-entered_x - 1, -1, -1):
            rh[column], latent[column] = extend_column(rh[column + 1], latent[column + 1])
    entered_z = int(movement[1])
    if entered_z > 0:
        for row in range(nz - entered_z, nz):
            rh[:, row], latent[:, row] = extend_row(rh[:, row - 1], latent[:, row - 1])
    elif entered_z < 0:
        for row in range(-entered_z - 1, -1, -1):
            rh[:, row], latent[:, row] = extend_row(rh[:, row + 1], latent[:, row + 1])
    return np.vstack([rh.reshape(cells, ne), latent.reshape(cells, ne)])


def engagement_invariant_windows(members, dims):
    """Per-member x-invariant engagement priors from each bit column.

    The 1D evidence constrains only the bit column; extending it laterally
    makes no untested lateral claims, unlike an ensemble whose flank
    realizations can sit orders of magnitude from 2D-data consistency.
    """
    nx, _, nz = dims
    values = np.asarray(members, dtype=float)
    if values.ndim != 2 or values.shape[0] != 2 * nx * nz:
        raise ValueError('Engagement members must have shape (2*nx*nz, members)')
    ne = values.shape[1]
    cells = nx * nz
    column = values.reshape(2, nx, nz, ne)[:, nx // 2, :, :]
    return np.vstack([np.repeat(column[field][:, None, :], nx, axis=1)
                      .transpose(1, 0, 2).reshape(cells, ne)
                      for field in (0, 1)])


def _build_param_dict(param):
    n_cells = dims[0] * dims[2]
    param = np.asarray(param, dtype=float)
    if param.shape != (2 * n_cells,):
        raise ValueError('2D parameter vector does not match the local window')
    log_rh = param[:n_cells].reshape(dims[0], dims[2])
    latent = param[n_cells:].reshape(dims[0], dims[2])
    log_rv = log_rv_from_latent(log_rh, latent, min_ratio, max_ratio, log_rv_cap, anisotropy)
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
    log_rv = log_rv_from_latent(log_rh, latent, min_ratio, max_ratio, log_rv_cap, anisotropy)
    parameters = {'rh': log_rh.item() if size == 1 else log_rh,
                  'rv': log_rv.item() if size == 1 else log_rv}
    sim = UTA0D_sim if simfidelity == '0D' else UTA1D_sim
    prediction = sim.run_fwd_sim(parameters, 0)
    vector = np.concatenate([np.asarray(prediction[0][tool]).ravel() for tool in tools])
    if vector.shape != (8 * len(tools),) or not np.isfinite(vector).all():
        raise RuntimeError(f'Invalid {simfidelity} UDAR response')
    return vector


def simulate_and_grad(param, logging_point_idx=0):
    parameters, log_rh, latent = _build_param_dict(param)
    prediction, j_rh, j_rv = remote_2d.run_grad(parameters, logging_point_idx, current_shift, wellpath)
    return prediction, optimizer_jacobian(j_rh, j_rv, latent, min_ratio, max_ratio,
                                           log_rv_cap=log_rv_cap, log_rh=log_rh,
                                           anisotropy=anisotropy)


def simulate_and_grad_multi(param_list, logging_point_idx=0):
    """Batched 2D evaluation; one (prediction, jacobian) or Exception per input."""
    param_dicts = [_build_param_dict(param) for param in param_list]
    results = remote_2d.run_grad_multi([p[0] for p in param_dicts],
                                       logging_point_idx, current_shift, wellpath)
    outcome = []
    for (parameters, log_rh, latent), (prediction, j_rh, j_rv) in zip(param_dicts, results):
        if j_rh is None:
            outcome.append(RuntimeError('2D evaluation failed for one member field'))
            continue
        outcome.append((prediction, optimizer_jacobian(j_rh, j_rv, latent, min_ratio, max_ratio,
                                                       log_rv_cap=log_rv_cap, log_rh=log_rh,
                                                       anisotropy=anisotropy)))
    return outcome


def custom_loss(predictions, data_real, theta, theta_mean, C_theta, Cd):
    residual = np.asarray(data_real).ravel() - np.asarray(predictions).ravel()
    prior_residual = np.asarray(theta).ravel() - np.asarray(theta_mean).ravel()
    prior_covariance = np.asarray(C_theta)
    if prior_covariance.ndim == 1:
        prior_loss = np.sum(prior_residual ** 2 / prior_covariance)
    else:
        prior_loss = prior_residual @ np.linalg.solve(prior_covariance, prior_residual)
    return 0.5 * (np.sum(residual ** 2 / Cd) + prior_loss)


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


def _auto_adaloc_config(localization_threshold, cells):
    """PET's sigmoid midpoint is ``1 - autoadaloc``; encode the requested cutoff."""
    if not np.isfinite(localization_threshold) or not 0 < localization_threshold < 1:
        raise ValueError('Require 0 < localization threshold < 1')
    return {'field': [int(cells), 1, 1],
            'autoadaloc': round(1. - float(localization_threshold), 12), 'type': 'sigm'}


def _configure_auto_adaloc(updater, localization_threshold, cells, member_count):
    from pipt.misc_tools.cov_regularization import localization

    config = _auto_adaloc_config(localization_threshold, cells)
    updater.keys_da = {'localization': config}
    updater.localization = localization(config, [], [], updater.list_states, member_count)
    return config


def _paired_mean_correction(ensemble_1d, selected_members, selected_1d, selected_2d,
                            rh_cells, rh_limits=LOG_RH_LIMITS):
    """Carry the paired 2D-minus-1D mean shift into members not 2D-updated.

    A common per-coordinate shift leaves the remaining members' internal
    covariance unchanged. Rh-cell shifts are damped only as needed to keep all
    transferred members inside physical bounds.
    """
    ensemble_1d = np.asarray(ensemble_1d, dtype=float)
    selected_members = np.asarray(selected_members, dtype=int)
    selected_1d = np.asarray(selected_1d, dtype=float)
    selected_2d = np.asarray(selected_2d, dtype=float)
    if (ensemble_1d.ndim != 2 or selected_1d.shape != selected_2d.shape
            or selected_1d.shape != (ensemble_1d.shape[0], selected_members.size)
            or rh_cells != ensemble_1d.shape[0] // 2):
        raise ValueError('Paired ensemble states have incompatible dimensions')
    if selected_members.size < 1 or np.unique(selected_members).size != selected_members.size:
        raise ValueError('Require unique selected member indices')

    correction = np.mean(selected_2d - selected_1d, axis=1)
    transfer_members = np.ones(ensemble_1d.shape[1], dtype=bool)
    transfer_members[selected_members] = False
    damping = np.ones(ensemble_1d.shape[0], dtype=float)
    if transfer_members.any():
        rh = ensemble_1d[:rh_cells, transfer_members]
        rh_shift = correction[:rh_cells]
        positive = rh_shift > 0
        negative = rh_shift < 0
        if positive.any():
            rows = np.flatnonzero(positive)
            row_max = np.max(rh[rows], axis=1)
            damping[rows] = np.minimum(
                1., np.maximum(0., (rh_limits[1] - row_max) / rh_shift[rows]))
        if negative.any():
            rows = np.flatnonzero(negative)
            row_min = np.min(rh[rows], axis=1)
            damping[rows] = np.minimum(
                1., np.maximum(0., (rh_limits[0] - row_min) / rh_shift[rows]))

    corrected = ensemble_1d.copy()
    corrected[:, transfer_members] += damping[:, None] * correction[:, None]
    corrected[:, selected_members] = selected_2d
    return corrected, correction, damping


def _run_two_step_esmda(state, member_indices, observations, data_covariance,
                        remaining_weight, forward_batch, update_step, rng,
                        localization_threshold, post_update=None,
                        rh_limits=LOG_RH_LIMITS):
    """Run two forward-only ES-MDA updates on selected members, dropping failed runs."""
    member_indices = np.asarray(member_indices, dtype=int)
    if member_indices.size < 2 or not 0 < remaining_weight <= 1:
        raise ValueError('Two-step ES-MDA requires at least two members and positive remaining weight')
    alpha = 2. / remaining_weight
    subset = np.asarray(state, dtype=float)[:, member_indices].copy()
    failures = []

    def evaluate(parameters, step):
        predictions = forward_batch([parameters[:, i].copy()
                                     for i in range(parameters.shape[1])])
        if len(predictions) != parameters.shape[1]:
            raise ValueError('forward_batch must return one result per member')
        valid = []
        for slot, prediction in enumerate(predictions):
            if prediction is None or isinstance(prediction, Exception):
                failures.append({'step': step, 'member': int(member_indices[slot]),
                                 'error': str(prediction)})
                continue
            prediction = np.asarray(prediction, dtype=float).ravel()
            if prediction.shape != np.asarray(observations).ravel().shape or not np.isfinite(prediction).all():
                failures.append({'step': step, 'member': int(member_indices[slot]),
                                 'error': 'invalid or non-finite forward prediction'})
                continue
            valid.append((slot, prediction))
        if len(valid) < 2:
            return None, None, None
        keep = np.asarray([slot for slot, _ in valid], dtype=int)
        return (parameters[:, keep], np.column_stack([prediction for _, prediction in valid]),
                member_indices[keep])

    current = evaluate(subset, 'initial')
    if current[0] is None:
        return None
    subset, prediction, member_indices = current
    alpha_history = []
    for step in (1, 2):
        inflated = np.asarray(data_covariance, dtype=float) * alpha
        perturbed = rng.normal(np.asarray(observations)[:, None], np.sqrt(inflated)[:, None],
                               (len(inflated), subset.shape[1]))
        delta = update_step(subset, prediction, perturbed, inflated, localization_threshold)
        subset += delta
        active = subset.shape[0] // 2
        subset[:active] = np.clip(subset[:active], *rh_limits)
        if post_update is not None:
            subset = post_update(subset, alpha)
        alpha_history.append(float(alpha))
        evaluated = evaluate(subset, step)
        if evaluated[0] is None:
            return None
        subset, prediction, member_indices = evaluated
    result = np.asarray(state, dtype=float).copy()
    result[:, member_indices] = subset
    return {'state': result, 'member_indices': member_indices,
            'parameters': subset, 'predictions': prediction,
            'alpha': alpha_history, 'failures': failures,
            'inflation_factor': alpha}


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
    parser.add_argument('--resume-checkpoint', type=Path, default=None,
                        help='Warm-start from the checkpoint at --start minus one')
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
    parser.add_argument('--engagement-prior', choices=['ensemble'], default='ensemble',
                        help='Use the carried 2D realization for final-stage engagement')
    parser.add_argument('--max-lm-trials', type=int, default=2)
    parser.add_argument('--misfit-threshold', type=float, default=4.,
                        help='Reduced measurement misfit below which 1D is deemed sufficient')
    parser.add_argument('--one-d-budget', type=float, default=.75,
                        help='Maximum ES-MDA weight the 1D stage may spend before the final '
                             'stage; the remainder is reserved for the final likelihood '
                             '(0D always spends up to .25)')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--force-2d', action='store_true', help='Exercise 2D even if the lower fidelities converge')
    parser.add_argument('--single-update', action='store_true',
                        help='Engaged 2D points take one batched linearized update '
                             'instead of iterated LM (one solver batch per point)')
    parser.add_argument('--final-stage', choices=['legacy', 'lm-laplace', 'two-step-esmda'],
                        default='legacy')
    parser.add_argument('--final-2d-members', type=int, default=10,
                        help='Best 1D-misfit members used by the two-step 2D ES-MDA stage')
    parser.add_argument('--adaptive-localization-threshold', '--final-2d-localization-threshold',
                        dest='final_2d_localization_threshold', type=float, default=.8,
                        help='Correlation midpoint for sigmoid autoADAloc in 1D and final 2D updates')
    parser.add_argument('--lm-laplace-iterations', type=int, default=12)
    parser.add_argument('--lm-laplace-starts', type=int, default=5,
                        help='Number of best 1D-ranked starts to optimize in 2D before Laplace sampling')
    parser.add_argument('--lm-laplace-seconds', type=float, default=1200.)
    parser.add_argument('--lm-relative-data-improvement', type=float, default=.05,
                        help='Converge when accepted data-misfit improvement is below this fraction')
    parser.add_argument('--fd-step', type=float, default=1e-3)
    parser.add_argument('--prior-style', choices=['blocky', 'gaussian', 'mixture'], default='blocky',
                        help='Prior family: legacy blocky, smooth Gaussian, or long-block mixture')
    parser.add_argument('--prior-background', type=float, default=float(np.log10(1.5)),
                        help='log10 Rh of the background mode (Ohm-m)')
    parser.add_argument('--prior-reservoir', type=float, default=float(np.log10(30.)),
                        help='log10 Rh of the reservoir mode (Ohm-m)')
    parser.add_argument('--prior-mode-std', type=float, default=0.10,
                        help='Within-mode log10 Rh standard deviation')
    parser.add_argument('--prior-reservoir-fraction', type=float, default=0.15,
                        help='Fraction of reservoir-mode cells in the prior')
    parser.add_argument('--prior-switch-prob', type=float, default=0.05,
                        help='Legacy blocky prior per-cell mode-flip probability')
    parser.add_argument('--prior-min-block-ft', type=float, default=50.,
                        help='Minimum ordinary mixture-block length, feet')
    parser.add_argument('--prior-fault-fraction', type=float, default=.15,
                        help='Mixture weight for the short-block/fault component')
    parser.add_argument('--prior-fault-min-block-ft', type=float, default=15.,
                        help='Minimum fault-component block length, feet')
    parser.add_argument('--prior-within-block-std', type=float, default=.02,
                        help='Correlated within-block lnRh residual standard deviation in log10 units')
    parser.add_argument('--prior-interface-jitter', type=float, default=.01,
                        help='Per-column probability of moving a whole mixture interface by one cell')
    parser.add_argument('--prior-value-correlation', type=float, default=.98,
                        help='Mean-reverting correlation for block values across adjacent columns')
    parser.add_argument('--prior-correlation-x-ft', type=float, default=10.,
                        help='Gaussian prior lateral correlation length in feet (ln Rh)')
    parser.add_argument('--prior-latent-correlation-x-ft', type=float, default=80.,
                        help='Gaussian prior lateral correlation length in feet (anisotropy latent)')
    parser.add_argument('--prior-correlation-z-ft', type=float, default=75.,
                        help='Gaussian prior vertical correlation length in feet (ln Rh)')
    parser.add_argument('--prior-latent-correlation-z-ft', type=float, default=35.,
                        help='Gaussian prior vertical correlation length in feet (latent/ratio)')
    parser.add_argument('--prior-ratio-max', type=float, default=3.,
                        help='Maximum allowed Rv/Rh ratio before the physical Rv cap')
    parser.add_argument('--prior-rv-cap-ohmm', type=float, default=80.,
                        help='Hard physical cap applied to Rv in all simulator fidelities')
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
    if args.final_2d_members < 2:
        parser.error('Require final-2d-members >= 2')
    if (not np.isfinite(args.final_2d_localization_threshold)
            or not 0 < args.final_2d_localization_threshold < 1):
        parser.error('Require 0 < final-2d-localization-threshold < 1')
    if args.final_stage == 'lm-laplace':
        if args.regularization != 'none' or args.prior_style != 'gaussian':
            parser.error('LM/Laplace validation requires Gaussian prior and regularization none')
        if (args.lm_laplace_iterations < 1 or args.lm_laplace_starts < 1
                or not np.isfinite(args.lm_laplace_seconds)
                or args.lm_laplace_seconds <= 0 or not np.isfinite(args.fd_step) or args.fd_step <= 0):
            parser.error('Require positive LM start/iteration/time budgets and FD step')
        if (not np.isfinite(args.lm_relative_data_improvement)
                or not 0 < args.lm_relative_data_improvement < 1):
            parser.error('Require 0 < lm-relative-data-improvement < 1')
    if args.ensemble_size < 2 or not 1 <= args.opt_members <= args.ensemble_size:
        parser.error('Require ensemble-size >= 2 and 1 <= opt-members <= ensemble-size')
    if args.final_stage == 'two-step-esmda' and args.sim_backend == 'slurm':
        args.concurrency = max(args.concurrency,
                               min(args.final_2d_members, args.ensemble_size))
    if min(args.frequency, args.nx, args.nz, args.y_layers, args.max_lm_trials) < 1 or args.max_iterations < 0:
        parser.error('Sizes, frequency and LM trials must be positive; iterations must be nonnegative')
    if args.max_substitutions < 0:
        parser.error('Require max-substitutions >= 0')
    if not np.isfinite([args.dx_ft, args.dz_ft]).all() or min(args.dx_ft, args.dz_ft) <= 0:
        parser.error('Cell sizes must be positive and finite')
    if not np.isfinite(args.misfit_threshold) or args.misfit_threshold <= 0:
        parser.error('Misfit threshold must be positive and finite')
    if not np.isfinite(args.one_d_budget) or not 0 < args.one_d_budget <= .75:
        parser.error('The 1D budget must lie in (0, .75] so the 0D and 1D stages '
                     'cannot overspend the pre-final weight')
    if args.start < 0 or (args.stop is not None and args.stop <= args.start):
        parser.error('Require 0 <= start < stop')
    if args.resume_checkpoint is not None and args.start < 1:
        parser.error('Resume checkpoint requires start >= 1')
    if args.resume_checkpoint is not None and args.start < 1:
        parser.error('Resume checkpoint requires start >= 1')
    if not np.isfinite([args.prior_background, args.prior_reservoir, args.prior_mode_std,
                        args.prior_reservoir_fraction, args.prior_switch_prob]).all():
        parser.error('Prior parameters must be finite')
    if not np.isfinite([args.prior_min_block_ft, args.prior_fault_fraction,
                        args.prior_within_block_std,
                        args.prior_interface_jitter, args.prior_value_correlation,
                        args.prior_correlation_x_ft, args.prior_correlation_z_ft,
                        args.prior_latent_correlation_x_ft,
                        args.prior_latent_correlation_z_ft, args.prior_ratio_max,
                        args.prior_rv_cap_ohmm]).all():
        parser.error('Spatial prior parameters must be finite')
    if not args.prior_background < args.prior_reservoir:
        parser.error('Require prior-background < prior-reservoir (log10 Ohm-m)')
    if min(args.prior_mode_std, args.prior_reservoir_fraction) <= 0 \
            or args.prior_reservoir_fraction >= 1 or not 0 <= args.prior_switch_prob < 1:
        parser.error('Require 0 < mode-std, 0 < reservoir-fraction < 1, 0 <= switch-prob < 1')
    if (args.prior_min_block_ft <= 0 or args.prior_fault_min_block_ft <= 0
            or args.prior_within_block_std < 0
            or not 0 <= args.prior_fault_fraction <= 1
            or not 0 <= args.prior_interface_jitter < 1
            or not 0 <= args.prior_value_correlation < 1
            or min(args.prior_correlation_x_ft, args.prior_latent_correlation_x_ft,
                  args.prior_correlation_z_ft,
                  args.prior_latent_correlation_z_ft, args.prior_rv_cap_ohmm) <= 0
            or args.prior_ratio_max <= min_ratio):
        parser.error('Invalid spatial prior scales/probabilities')
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


def load_resume_state(checkpoint_path, dims, spacing_ft, start, ensemble_size, tools,
                      prior_style, ratio_bounds, rv_cap_ohmm, prior_lengths,
                      window, wx, wy, tvd):
    """Load the previous posterior and reconstruct its moving-window offset."""
    checkpoint_path = Path(checkpoint_path).resolve()
    raw = checkpoint_path.read_bytes()
    saved = pickle.loads(raw)
    if saved.get('logging_index') != start - 1:
        raise ValueError('Resume checkpoint must be the immediately preceding logging index')
    if tuple(saved.get('dims', ())) != tuple(dims):
        raise ValueError('Resume checkpoint dimensions do not match this run')
    if not np.allclose(saved.get('spacing_ft', ()), spacing_ft, rtol=0., atol=1e-10):
        raise ValueError('Resume checkpoint spacing does not match this run')
    if [tuple(tool) for tool in saved.get('tools', [])] != [tuple(tool) for tool in tools]:
        raise ValueError('Resume checkpoint tools do not match the current observations')
    if saved.get('prior_style') != prior_style:
        raise ValueError('Resume checkpoint prior family does not match this run')
    if tuple(saved.get('rv_ratio_bounds', ())) != tuple(ratio_bounds):
        raise ValueError('Resume checkpoint anisotropy ratio bounds do not match this run')
    if not np.isclose(saved.get('rv_hard_cap_ohmm', np.nan), rv_cap_ohmm):
        raise ValueError('Resume checkpoint Rv cap does not match this run')
    saved_lengths = np.asarray(saved.get('laplace_correlation_ft'), dtype=float)
    if saved_lengths.shape != (2, 2) or not np.allclose(saved_lengths, prior_lengths):
        raise ValueError('Resume checkpoint spatial prior lengths do not match this run')
    params = np.asarray(saved.get('posterior_ensemble_params',
                                 saved.get('posterior_params')), dtype=float)
    expected_params = 2 * dims[0] * dims[2]
    if params.shape != (ensemble_size, expected_params) or not np.isfinite(params).all():
        raise ValueError('Resume checkpoint posterior ensemble has invalid shape or values')
    if start > len(tvd):
        raise ValueError('Resume point lies outside the well trajectory')
    shift = None
    for index in range(start):
        shift, _ = window.update((wx[index], wy[index], tvd[index]))
    saved_shift = saved.get('shift_ft', {})
    if not all(np.isclose(shift[key], saved_shift[key], rtol=0., atol=1e-6)
               for key in ('x', 'y', 'z')):
        raise ValueError('Reconstructed moving-window origin differs from the checkpoint')
    return params.T.copy(), saved['logging_index'], hashlib.sha256(raw).hexdigest()


def main(argv=None):
    global dims, dX, dY, dZ, tools, current_shift, remote_2d
    global max_ratio, rv_cap_ohmm, log_rv_cap
    args, config = parse_args(argv)
    max_ratio = args.prior_ratio_max
    rv_cap_ohmm = args.prior_rv_cap_ohmm
    log_rv_cap = float(np.log(rv_cap_ohmm))
    tv_weight_1d = args.tv_weight if args.tv_weight_1d is None else args.tv_weight_1d
    tv_weight_2d = args.tv_weight if args.tv_weight_2d is None else args.tv_weight_2d
    background, reservoir, mode_std = prior_parameters_ln(args)
    projection = (args.prior_style == 'blocky' and args.prior_projection == 'on'
                  and args.regularization != 'tv')
    from pipt.update_schemes.update_methods_ns.approx_update import approx_update
    benchmark = args.benchmark.resolve()
    resume_checkpoint = (args.resume_checkpoint.resolve()
                         if args.resume_checkpoint is not None else None)
    effective_random_seed = args.seed if resume_checkpoint is None else args.seed + args.start
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
    np.random.seed(effective_random_seed)
    try:
        os.chdir(output)
        setup_simulators(benchmark, tools)
        remote_2d = make_remote2d(config, dims, dX, dY, dZ, tools)
        remote_2d.jobs_dir = str(output / 'jobs')
        if not remote_2d.backend().trusts_ad('2D'):
            raise RuntimeError('Select an exact-adjoint 2D solver build')
        if resume_checkpoint is None:
            previous, previous_index, resume_checkpoint_sha256 = None, None, None
        else:
            prior_lengths = ((args.prior_correlation_x_ft, args.prior_correlation_z_ft),
                             (args.prior_latent_correlation_x_ft,
                              args.prior_latent_correlation_z_ft))
            previous, previous_index, resume_checkpoint_sha256 = load_resume_state(
                resume_checkpoint, dims, (dX, dY, dZ), args.start, args.ensemble_size,
                tool_pairs, args.prior_style, (min_ratio, max_ratio), rv_cap_ohmm,
                prior_lengths, window, WX, WY, TVD)
        manifest = {**vars(args), 'benchmark': benchmark, 'simulator': asdict(config),
                    'runtime': remote_2d.runtime_identity(),
                    'resume_checkpoint': resume_checkpoint,
                    'resume_checkpoint_sha256': resume_checkpoint_sha256,
                    'effective_random_seed': effective_random_seed,
                    'tools': tool_pairs, 'dims': dims, 'spacing_ft': (dX, dY, dZ),
                     'covariance_bytes': (2 * args.nx * args.nz) ** 2 * 8,
                      'parameter_units': '[ln(Rh / Ohm m), anisotropy latent]',
                      'rv_parameterization': {'ratio_bounds': [min_ratio, max_ratio],
                                              'latent_std': ratio_latent_std,
                                              'hard_cap_ohmm': rv_cap_ohmm},
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
                if fidelity == '1D':
                    _configure_auto_adaloc(updater, args.final_2d_localization_threshold,
                                           cells, state.shape[1])
                else:
                    updater.keys_da = {}
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

            def final_1d_update(source_state, source_prediction, factor):
                state_1d = np.asarray(source_state, dtype=float).copy()
                active = state_1d.shape[0] // 2
                updater.state_scaling = np.ones(state_1d.shape[0])
                updater.prior_info = {key: {'active': active} for key in ('rh', 'rv')}
                _configure_auto_adaloc(updater, args.final_2d_localization_threshold,
                                       cells, state_1d.shape[1])
                inflated = Cd * factor
                updater.scale_data = np.sqrt(inflated)
                perturbed = np.random.normal(observations[:, None], np.sqrt(inflated)[:, None],
                                             (len(Cd), state_1d.shape[1]))
                updater.update(enX=state_1d, enY=source_prediction, enE=perturbed)
                state_1d += updater.step
                np.clip(state_1d[:active], *LOG_RH_LIMITS, out=state_1d[:active])
                if projection:
                    state_1d[:active] = project_to_modes(
                        state_1d[:active], background, reservoir, 2. * mode_std)
                if args.regularization == 'tv':
                    final_fraction = 1 / factor / spatial_weight[0]
                    dose = tv_weight_1d * final_fraction
                    state_1d = tv.regularize_windows(
                        state_1d, nx, nz, dose, dX, dZ, args.tv_epsilon, args.tv_iterations)
                prediction_1d = predict_state(state_1d, '1D')
                final_misfit = measurement_misfit(prediction_1d.mean(axis=1), observations, Cd)
                return state_1d, prediction_1d, perturbed, final_misfit

            if previous is None:
                state = clamp_log_rh(sample_prior_0D(ne, background, reservoir, mode_std,
                                                     args.prior_reservoir_fraction)[0])
                state, _ = assimilate_stage(state, '0D', .25)
                state = clamp_log_rh(sample_prior_2d(ne, state[0], background, reservoir, mode_std,
                                                     args.prior_reservoir_fraction,
                                                     args.prior_switch_prob, projection=projection,
                                                     prior_style=args.prior_style, dx=dX, dz=dZ,
                                                     min_block_cells=max(1, int(np.ceil(args.prior_min_block_ft / dZ))),
                                                     fault_fraction=args.prior_fault_fraction,
                                                     fault_min_block_cells=max(1, int(np.ceil(
                                                         args.prior_fault_min_block_ft / dZ))),
                                                     within_block_std=np.log(10.) * args.prior_within_block_std,
                                                     interface_jitter=args.prior_interface_jitter,
                                                     value_correlation=args.prior_value_correlation,
                                                     corr_x_ft=args.prior_correlation_x_ft,
                                                     corr_z_ft=args.prior_correlation_z_ft,
                                                     latent_corr_x_ft=args.prior_latent_correlation_x_ft,
                                                     latent_corr_z_ft=args.prior_latent_correlation_z_ft)[0])
                point_prior = state.copy()
                spatial_weight[0] = 1. - used_weight[0]
                state, prediction = assimilate_stage(state, '1D', args.one_d_budget)
            else:
                state = clamp_log_rh(shift_window_state(previous, movement, background, reservoir,
                                                        mode_std, ratio_latent_std,
                                                        args.prior_switch_prob,
                                                        prior_style=args.prior_style,
                                                        fraction=args.prior_reservoir_fraction,
                                                        dx=dX, dz=dZ,
                                                        min_block_cells=max(1, int(np.ceil(
                                                            args.prior_min_block_ft / dZ))),
                                                        fault_fraction=args.prior_fault_fraction,
                                                        fault_min_block_cells=max(1, int(np.ceil(
                                                            args.prior_fault_min_block_ft / dZ))),
                                                        within_block_std=np.log(10.) * args.prior_within_block_std,
                                                        interface_jitter=args.prior_interface_jitter,
                                                        value_correlation=args.prior_value_correlation,
                                                        corr_x_ft=args.prior_correlation_x_ft,
                                                        corr_z_ft=args.prior_correlation_z_ft,
                                                        latent_corr_x_ft=args.prior_latent_correlation_x_ft,
                                                        latent_corr_z_ft=args.prior_latent_correlation_z_ft))
                point_prior = state.copy()
                state, _ = assimilate_stage(state, '0D', .25)
                spatial_weight[0] = 1. - used_weight[0]
                state, prediction = assimilate_stage(state, '1D', args.one_d_budget)
            post_1d_state = state.copy()
            final_1d_baseline = None
            if args.final_stage == 'two-step-esmda':
                gate_open = args.force_2d or misfit['1D_final'] > threshold
                if gate_open:
                    factor = 1. / (1. - used_weight[0])
                    final_1d_baseline = final_1d_update(post_1d_state, prediction, factor)
                    baseline_state, baseline_prediction, _, baseline_misfit = final_1d_baseline
                    n_final_members = min(args.final_2d_members, ne)
                    ranked_1d = np.argsort([
                        measurement_misfit(prediction[:, member], observations, Cd)
                        for member in range(ne)])
                    selected = ranked_1d[:n_final_members]
                    calls = {'2D': 0}

                    def forward_2d_batch(models):
                        calls['2D'] += len(models)
                        param_dicts = [_build_param_dict(model)[0] for model in models]
                        predictions_2d = remote_2d.run_pred_multi(
                            param_dicts, index, current_shift, wellpath)
                        return [pred if isinstance(pred, Exception)
                                else (np.asarray(pred)[obs_mask]
                                      if pred is not None and np.isfinite(pred).all() else None)
                                for pred in predictions_2d]

                    def localized_es_mda_step(enX, enY, enE, inflated, loc_threshold):
                        member_count = enX.shape[1]
                        updater = approx_update()
                        updater.proj = (np.eye(member_count)
                                        - np.ones((member_count, member_count)) / member_count) \
                            / np.sqrt(member_count - 1)
                        updater.trunc_energy = .95
                        updater.lam = 0
                        updater.list_states = ['rh', 'rv']
                        updater.state_scaling = np.ones(enX.shape[0])
                        updater.prior_info = {key: {'active': cells} for key in ('rh', 'rv')}
                        updater.scale_data = np.sqrt(inflated)
                        _configure_auto_adaloc(
                            updater, loc_threshold, cells, member_count)
                        updater.update(enX=enX, enY=enY, enE=enE)
                        return updater.step

                    def constrain_final_2d(subset, alpha):
                        if projection:
                            subset[:cells] = project_to_modes(
                                subset[:cells], background, reservoir, 2. * mode_std)
                        if args.regularization == 'tv':
                            dose = tv_weight_2d / alpha / spatial_weight[0]
                            subset = tv.regularize_windows(
                                subset, nx, nz, dose, dX, dZ, args.tv_epsilon,
                                args.tv_iterations)
                            tv_history.append({'stage': 'final_2D_es_mda', 'strength': dose,
                                               'assimilation_fraction': 1 / alpha /
                                               spatial_weight[0]})
                        return subset

                    tv_history_mark = len(tv_history)
                    two_step = _run_two_step_esmda(
                        post_1d_state, selected, observations, Cd, 1. / factor,
                        forward_2d_batch, localized_es_mda_step, np.random,
                        args.final_2d_localization_threshold, constrain_final_2d)
                    if two_step is not None:
                        lower_fidelity_weight = used_weight[0]
                        state, mean_correction, correction_damping = _paired_mean_correction(
                            baseline_state, two_step['member_indices'],
                            baseline_state[:, two_step['member_indices']],
                            two_step['parameters'], cells)
                        used_weight[0] += 1. / factor
                        final_misfits = [measurement_misfit(two_step['predictions'][:, member],
                                                            observations, Cd)
                                         for member in range(two_step['predictions'].shape[1])]
                        posterior_predictions = [pred.copy()
                                                 for pred in two_step['predictions'].T]
                        misfit.update(before_final=misfit['1D_final'],
                                      after_final=float(np.mean(final_misfits)),
                                      after_final_kind='forward')
                        results = {
                            'prior_params': point_prior,
                            'post_mda_params': post_1d_state.copy(),
                            'post_1d_params': post_1d_state.copy(),
                            'posterior_params': [params.copy()
                                                 for params in two_step['parameters'].T],
                            'posterior_predictions': posterior_predictions,
                            'posterior_losses': final_misfits,
                            'optimizer_history': [], 'trial_errors': [],
                            'observations': observations,
                            'data_covariance': Cd * two_step['inflation_factor'],
                            'posterior_ensemble_params': [params.copy() for params in state.T],
                            'mean_correction_method': 'paired_selected_2d_minus_1d',
                            'mean_correction_delta': mean_correction,
                            'mean_correction_damping': correction_damping,
                            'mean_correction_damping_min_rh': float(
                                correction_damping[:cells].min()),
                            'mean_correction_damping_mean_rh': float(
                                correction_damping[:cells].mean()),
                            'mean_correction_norm': float(np.linalg.norm(mean_correction)),
                            'final_1d_baseline_misfit': baseline_misfit,
                            'dims': dims, 'spacing_ft': (dX, dY, dZ),
                            'shift_ft': dict(current_shift), 'window_movement': movement,
                            'previous_index': previous_index, 'logging_index': index,
                            'tools': tool_pairs, 'seed': args.seed,
                            'effective_random_seed': effective_random_seed,
                            'resume_checkpoint_sha256': resume_checkpoint_sha256,
                            'mda_weight': 1., 'lower_fidelity_weight': lower_fidelity_weight,
                            'final_inflation_factor': two_step['inflation_factor'],
                            'optimized_2d': True, 'selected_fidelity': '2D',
                            'final_stage': 'two-step-esmda', 'two_d_mode': 'jacobian-free-es-mda',
                            'two_d_calls': calls['2D'],
                            'simulator_calls': {'1D': 0, '2D': calls['2D'],
                                                'jacobian_2D': 0},
                            'member_candidates': two_step['member_indices'].tolist(),
                            'final_2d_ranked_candidates': selected.tolist(),
                            'final_2d_alpha': two_step['alpha'],
                            'final_2d_localization': {
                                'type': 'sigm',
                                'correlation_threshold': args.final_2d_localization_threshold,
                                'autoadaloc': _auto_adaloc_config(
                                    args.final_2d_localization_threshold, cells)['autoadaloc']},
                            'adaptive_localization': {
                                'type': 'sigm',
                                'correlation_threshold': args.final_2d_localization_threshold,
                                'autoadaloc': _auto_adaloc_config(
                                    args.final_2d_localization_threshold, cells)['autoadaloc']},
                            'solver_failures': two_step['failures'],
                            'substituted_members': 0,
                            'prior_style': args.prior_style,
                            'misfit': misfit, 'misfit_threshold': threshold,
                            'posterior_misfit': final_misfits,
                            'observation_mask': obs_mask,
                            'masked_channels': masked_channels,
                            'masked_components': masked_udar_components,
                            'unified_domain': True, 'parameter_units': 'natural_log',
                            'prior_projection': args.prior_projection,
                            'prior_projection_effective': projection,
                            'rv_ratio_bounds': (min_ratio, max_ratio),
                            'rv_hard_cap_ohmm': rv_cap_ohmm,
                            'prior_units': manifest['prior_units'],
                            'posterior_prediction_kind': 'forward',
                            'posterior_misfit_kind': 'forward',
                            'posterior_forward_verified': True,
                            'regularization': args.regularization,
                            'tv_weight': args.tv_weight, 'tv_epsilon': args.tv_epsilon,
                            'tv_iterations': args.tv_iterations, 'tv_history': tv_history,
                            'tv_weights_effective': manifest['tv_weights_effective'],
                            'point_elapsed_seconds': time.monotonic() - point_started}
                        with open(f'inversion_results_assim_{index}.pkl', 'xb') as stream:
                            pickle.dump(results, stream)
                        previous, previous_index = state, index
                        print(f'Index {index}, two-step 2D ES-MDA: '
                              f'misfit={misfit["before_final"]:.6g} -> '
                              f'{misfit["after_final"]:.6g}, members='
                              f'{len(two_step["member_indices"])}, mean-shift scale='
                              f'{correction_damping[:cells].min():.3g}–'
                              f'{correction_damping[:cells].mean():.3g}, calls={calls["2D"]}',
                              flush=True)
                        continue
                    del tv_history[tv_history_mark:]
                    print(f'Index {index}: two-step 2D ES-MDA failed '
                          '(fewer than two members had valid forward predictions); '
                          'using final 1D update', flush=True)
            if args.final_stage == 'lm-laplace':
                from lm_laplace import final_stage
                calls = {'1D': 0, '2D': 0, 'jacobian_2D': 0}
                phase_mask = (np.arange(len(obs_mask)) % 2 == 0)[obs_mask]
                column_indices = np.r_[np.arange(bit * nz, (bit + 1) * nz),
                                       np.arange(cells + bit * nz, cells + (bit + 1) * nz)]

                def forward_1d(model):
                    calls['1D'] += 1
                    return simulate_pred_only(model[column_indices], index, '1D')[obs_mask]

                def jacobian_1d(model):
                    jac = np.zeros((len(observations), len(model)))
                    for parameter in column_indices:
                        plus, minus = model.copy(), model.copy()
                        plus[parameter] += args.fd_step
                        minus[parameter] -= args.fd_step
                        difference = forward_1d(plus) - forward_1d(minus)
                        difference[phase_mask] = (difference[phase_mask] + 180.) % 360. - 180.
                        jac[:, parameter] = difference / (2 * args.fd_step)
                    return jac

                def forward_2d(model):
                    calls['2D'] += 1
                    return simulate_pred_only(model, index, '2D')[obs_mask]

                def jacobian_2d(model):
                    calls['jacobian_2D'] += 1
                    prediction_j, jac = simulate_and_grad(model, index)
                    return jac[obs_mask]

                def forward_2d_batch(models):
                    calls['2D'] += len(models)
                    param_dicts = [_build_param_dict(model)[0] for model in models]
                    predictions_2d = remote_2d.run_pred_multi(
                        param_dicts, index, current_shift, wellpath)
                    return [np.asarray(pred)[obs_mask]
                            if np.isfinite(pred).all() else None for pred in predictions_2d]

                def jacobian_2d_batch(models):
                    calls['jacobian_2D'] += len(models)
                    outcomes = simulate_and_grad_multi(models, index)
                    return [None if isinstance(result, Exception) else result[1][obs_mask]
                            for result in outcomes]

                factor = 1. / (1. - used_weight[0])
                state, results = final_stage(
                    state, prediction, observations, Cd, factor, (nx, nz), (dX, dZ),
                    ((args.prior_correlation_x_ft, args.prior_correlation_z_ft),
                     (args.prior_latent_correlation_x_ft,
                      args.prior_latent_correlation_z_ft)), threshold,
                    forward_1d, jacobian_1d, forward_2d, jacobian_2d, np.random,
                    max_iterations=args.lm_laplace_iterations,
                    max_seconds=args.lm_laplace_seconds, max_step=args.tv_max_step,
                    relative_data_improvement=args.lm_relative_data_improvement,
                    is_recoverable_2d_failure=classify_solver_start_failure,
                    max_2d_starts=args.lm_laplace_starts,
                    invariant_start=None, forward_2d_batch=forward_2d_batch,
                    jacobian_2d_batch=jacobian_2d_batch)
                misfit.update(before_final=misfit['1D_final'],
                              after_final=results['map_forward_misfit'], after_final_kind='forward')
                results.update(
                    final_stage='lm-laplace', unified_domain=True,
                    dims=dims, spacing_ft=(dX, dY, dZ), shift_ft=dict(current_shift),
                    logging_index=index, previous_index=previous_index, window_movement=movement,
                    tools=tool_pairs, seed=args.seed, effective_random_seed=effective_random_seed,
                    resume_checkpoint_sha256=resume_checkpoint_sha256,
                    observations=observations,
                    data_covariance=Cd * factor, misfit=misfit, misfit_threshold=threshold,
                    mda_weight=1., lower_fidelity_weight=used_weight[0],
                    final_inflation_factor=factor, prior_params=point_prior,
                    post_mda_params=post_1d_state, post_1d_params=post_1d_state,
                    parameter_units='natural_log', prior_units=manifest['prior_units'],
                    prior_style=args.prior_style, regularization='none',
                    rv_ratio_bounds=(min_ratio, max_ratio), rv_hard_cap_ohmm=rv_cap_ohmm,
                    lm_laplace_starts=args.lm_laplace_starts,
                    solver_failures=results['two_d_initial_failures'], substituted_members=0,
                    two_d_calls=calls['2D'] + calls['jacobian_2D'], simulator_calls=calls,
                    observation_mask=obs_mask, masked_channels=masked_channels,
                    point_elapsed_seconds=time.monotonic() - point_started)
                print(f'Index {index}, LM/Laplace {results["selected_fidelity"]}: '
                      f'MAP misfit={results["map_forward_misfit"]:.6g}, '
                      f'converged={results["map_converged"]}, '
                      f'reason={results["map_termination"]}', flush=True)
                with open(f'inversion_results_assim_{index}.pkl', 'xb') as stream:
                    pickle.dump(results, stream)
                previous, previous_index = state, index
                continue
            mean = state.mean(axis=1)
            covariance_diagonal = np.maximum(state.var(axis=1), 1e-8)
            covariance = (covariance_diagonal if args.regularization == 'tv'
                          else np.diag(covariance_diagonal))
            ranking_full = np.argsort([custom_loss(prediction[:, member], observations,
                                                   state[:, member], mean,
                                                   covariance_diagonal, Cd)
                                       for member in range(ne)])
            ranking = ranking_full[:args.opt_members]
            factor = 1 / (1 - used_weight[0])
            inflated = Cd * factor
            final_tv_fraction = 1 / factor / spatial_weight[0]
            final_tv_dose = tv_weight_2d * final_tv_fraction
            escalate = args.force_2d or misfit['1D_final'] > threshold
            if args.final_stage == 'two-step-esmda':
                escalate = False
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
                covariance_diagonal_2d = np.maximum(prior.var(axis=1), 1e-8)
                covariance_2d = (covariance_diagonal_2d if args.regularization == 'tv'
                                 else np.diag(covariance_diagonal_2d))
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
                if final_1d_baseline is None:
                    final_1d_baseline = final_1d_update(state, prediction, factor)
                state, prediction, perturbed, misfit_now = final_1d_baseline
                if args.regularization == 'tv':
                    final_dose = tv_weight_1d * final_tv_fraction
                    tv_history.append({'stage': 'final_1D', 'strength': final_dose,
                                       'assimilation_fraction': final_tv_fraction})
                used_weight[0] += 1 / factor
                misfit['after_final'] = misfit_now
                worsened = bool(misfit_now > misfit['before_final'])
                print(f'Index {index}, final 1D update: misfit {misfit["before_final"]:.6g} -> '
                      f'{misfit_now:.6g}{" (increased)" if worsened else ""}, weight=1.0000', flush=True)
                ranking = np.argsort([custom_loss(prediction[:, member], perturbed[:, member],
                                                  state[:, member], mean,
                                                  covariance_diagonal, inflated)
                                      for member in range(ne)])[:args.opt_members]
                member_candidates = [int(rank) for rank in ranking]
                records = [{'parameters': state[:, ranking[member]].copy(),
                            'prediction': prediction[:, ranking[member]],
                            'loss': float(custom_loss(prediction[:, ranking[member]],
                                                      perturbed[:, member],
                                                      state[:, ranking[member]], mean,
                                                      covariance_diagonal, inflated)),
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
                        'effective_random_seed': effective_random_seed,
                        'resume_checkpoint_sha256': resume_checkpoint_sha256,
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
            if args.final_stage == 'two-step-esmda':
                results['posterior_ensemble_params'] = [params.copy() for params in state.T]
                results['adaptive_localization'] = {
                    'type': 'sigm',
                    'correlation_threshold': args.final_2d_localization_threshold,
                    'autoadaloc': _auto_adaloc_config(
                        args.final_2d_localization_threshold, cells)['autoadaloc']}
            if args.final_stage == 'two-step-esmda' and not (args.force_2d or
                                                              misfit['1D_final'] > threshold):
                results['final_stage_skipped'] = '1D-to-2D threshold gate remained closed'
            results.update(unified_domain=True, parameter_units='natural_log',
                           prior_projection=args.prior_projection,
                           rv_ratio_bounds=(min_ratio, max_ratio), rv_hard_cap_ohmm=rv_cap_ohmm,
                           lm_laplace_starts=args.lm_laplace_starts,
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
