#!/usr/bin/env python3
"""Step 1 of the heavy-tailed prior test: 1D column inversions with a blocky truth (local, no HPC).

At each chosen logging point the truth is the benchmark column under the bit (layered, two-valued). The
data are its 1D response plus noise from the run's data error model, so the forward model is exact and
only the prior differs between the inversions of the column state [ln Rh (nz), anisotropy latent (nz)]:

  matern    the current prior: the bit-column marginal of the window's Matern prior (Gaussian)
  student   Student-t increments between vertical neighbours plus a weak Gaussian anchor on the level

Both are solved as a MAP (Levenberg-Marquardt, finite-difference Jacobian; the Student-t prior by
iterative reweighting, started from the Matern MAP) with a Gaussian posterior around it (Laplace; for
Student-t the Gaussian conditional on the final weights). With --rml N the Student-t prior is also
sampled by N randomised MAPs (perturbed data and anchor, started from prior-like random columns), which
lets the boundary positions vary between members.

Scores are log10 Rh against the truth, by distance to the nearest truth boundary in the column.

With --data real the measured (benchmark, 2D) data are inverted instead, so the 1D model error is
included; --points-from RUN then selects the points that chain accepted as 1D (every --every-th).
The printed "model error" is the chi2 per datum between the data and the 1D response of the truth column.

python heavy_tail_1d_test.py --points 30 180 10 --out compare-output/heavy-tail-1d
python heavy_tail_1d_test.py --data real --points-from runs/my-run-4 --every 3 --out compare-output/heavy-tail-1d-real
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_DA_sequential as sequential
from esmda_core import LocalWindow, align_phases, noise_floor_mask, observation_vectors, phase_channels
from esmda_remote2d import UDAR_COMPONENTS
from pilot_cascade_laplace_index34 import matern_reference
from sequential_matern_1d import RH_CAP_OHMM, RV_CAP_OHMM, directional_floor, model_error_factors

ROOT = Path(__file__).resolve().parent
LN10 = np.log(10.)
RH_LIMITS = (-1., float(np.log(RH_CAP_OHMM)))
CLASSES = ['at boundary', '1 cell away', '2 cells away', 'interior, resistive', 'interior, conductive']


# ----------------------------------------------------------------------
# Priors on one column
# ----------------------------------------------------------------------
class GaussianColumn:
    """Bit-column marginal of the window Matern prior, one dense block per state block."""

    def __init__(self, prior, nx, nz):
        cells, bit = nx * nz, nx // 2
        rows = bit * nz + np.arange(nz)
        self.mean = np.concatenate([prior.mean[rows], prior.mean[cells + rows]])
        blocks = []
        for index, block in enumerate(prior.blocks):
            unit = np.zeros((cells, nz))
            unit[rows, np.arange(nz)] = 1.
            colour = block.color(block.color(unit, transpose=True))[rows]          # C restricted to the column
            blocks.append(np.linalg.inv(.5 * (colour + colour.T)))
        self.Q = np.block([[blocks[0], np.zeros((nz, nz))], [np.zeros((nz, nz)), blocks[1]]])

    def penalty(self, model):
        offset = model - self.mean
        return .5 * offset @ self.Q @ offset

    def quadratic(self, model):
        """Gradient of the penalty and the (majorising) precision at ``model``."""
        return self.Q @ (model - self.mean), self.Q


class StudentColumn:
    """Student-t increments (dof, scale per block) with a weak Gaussian anchor N(mean, anchor^2) per cell."""

    def __init__(self, mean, nz, dof, scales, anchors):
        self.mean, self.nz, self.dof = np.asarray(mean, float), nz, float(dof)
        difference = np.diff(np.eye(nz), axis=0)
        self.D = np.block([[difference, np.zeros_like(difference)], [np.zeros_like(difference), difference]])
        self.scale = np.repeat(np.asarray(scales, float), nz - 1)
        self.anchor = np.repeat(1. / np.asarray(anchors, float) ** 2, nz)

    def penalty(self, model):
        delta = self.D @ model
        return (.5 * (self.dof + 1.) * np.sum(np.log1p(delta ** 2 / (self.dof * self.scale ** 2)))
                + .5 * np.sum(self.anchor * (model - self.mean) ** 2))

    def quadratic(self, model, anchor_mean=None):
        delta = self.D @ model
        weights = (self.dof + 1.) / (self.dof * self.scale ** 2 + delta ** 2)
        precision = self.D.T @ (weights[:, None] * self.D) + np.diag(self.anchor)
        centre = self.mean if anchor_mean is None else anchor_mean
        return self.D.T @ (weights * delta) + self.anchor * (model - centre), precision

    def sample(self, rng):
        """A prior-like blocky start: cumulative Student-t increments around the mean."""
        steps = rng.standard_t(self.dof, size=self.scale.shape) * self.scale
        column = np.concatenate([np.r_[0., np.cumsum(steps[:self.nz - 1])],
                                 np.r_[0., np.cumsum(steps[self.nz - 1:])]])
        column[:self.nz] -= column[self.nz // 2]
        column[self.nz:] -= column[self.nz + self.nz // 2]
        return self.mean + column


# ----------------------------------------------------------------------
# MAP by Levenberg-Marquardt with a finite-difference Jacobian
# ----------------------------------------------------------------------
def clip_model(model, nz):
    model = np.array(model, dtype=float, copy=True)
    model[:nz] = np.clip(model[:nz], *RH_LIMITS)
    return model


def jacobian(forward, model, base, step=5e-3):
    columns = []
    for index in range(model.size):
        moved = model.copy()
        moved[index] += step
        columns.append((forward(moved) - base) / step)
    return np.column_stack(columns)


def solve_map(forward, observations, variance, prior, start, nz, max_iterations=20, tolerance=1e-3,
              anchor_mean=None):
    """Minimise 0.5 r^T Cd^-1 r + prior penalty. Returns model, Gaussian posterior covariance and a record."""
    def objective(model, predicted):
        residual = (observations - predicted) / np.sqrt(variance)
        extra = 0.
        if anchor_mean is not None:                 # randomised anchor of an RML member
            extra = (.5 * np.sum(prior.anchor * (model - anchor_mean) ** 2)
                     - .5 * np.sum(prior.anchor * (model - prior.mean) ** 2))
        return .5 * residual @ residual + prior.penalty(model) + extra

    model = clip_model(start, nz)
    predicted = forward(model)
    value, damping, calls = objective(model, predicted), 1e-2, 1
    for iteration in range(1, max_iterations + 1):
        J = jacobian(forward, model, predicted)
        calls += model.size
        weighted = J / np.sqrt(variance)[:, None]
        gradient_prior, precision = (prior.quadratic(model, anchor_mean) if anchor_mean is not None
                                     else prior.quadratic(model))
        hessian = weighted.T @ weighted + precision
        gradient = weighted.T @ ((observations - predicted) / np.sqrt(variance)) - gradient_prior
        improved = False
        for _ in range(8):
            step = np.linalg.solve(hessian + damping * np.diag(np.diag(hessian)), gradient)
            trial = clip_model(model + step, nz)
            trial_predicted = forward(trial)
            calls += 1
            trial_value = objective(trial, trial_predicted)
            if trial_value < value:
                improved, drop = True, (value - trial_value) / max(value, 1e-12)
                model, predicted, value, damping = trial, trial_predicted, trial_value, max(damping / 3., 1e-6)
                break
            damping *= 5.
        if not improved or drop < tolerance:
            break
    J = jacobian(forward, model, predicted)
    weighted = J / np.sqrt(variance)[:, None]
    precision = (prior.quadratic(model, anchor_mean) if anchor_mean is not None else prior.quadratic(model))[1]
    covariance = np.linalg.inv(weighted.T @ weighted + precision)
    residual = (observations - predicted) / np.sqrt(variance)
    return model, covariance, {'iterations': iteration, 'objective': float(value), 'calls': calls + model.size,
                               'chi2_per_datum': float(residual @ residual / residual.size)}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def classify_column(truth):
    """Class per cell of a log10 Rh truth column, by distance (cells) to the nearest jump."""
    jump = np.abs(np.diff(truth)) > .1
    edge = np.zeros(truth.size, bool)
    edge[1:] |= jump
    edge[:-1] |= jump
    distance = np.full(truth.size, 99)
    for cell in np.flatnonzero(edge):
        distance = np.minimum(distance, np.abs(np.arange(truth.size) - cell))
    return np.where(distance <= 2, distance, np.where(truth > 1., 3, 4))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, default=ROOT / 'runs/benchmark3-reference-bedding')
    parser.add_argument('--out', type=Path, default=ROOT / 'compare-output/heavy-tail-1d')
    parser.add_argument('--points', type=int, nargs=3, default=(30, 180, 10), metavar=('FIRST', 'LAST', 'STEP'))
    parser.add_argument('--data', choices=['twin', 'real'], default='twin',
                        help="twin: 1D response of the truth column plus noise; real: the measured data")
    parser.add_argument('--points-from', type=Path, default=None,
                        help='Chain run: use the logging points it accepted as 1D, within FIRST..LAST of --points')
    parser.add_argument('--every', type=int, default=1, help='With --points-from: keep every n-th point')
    parser.add_argument('--dof', type=float, default=1.)
    parser.add_argument('--scale', type=float, nargs=2, default=(.1, .05), metavar=('RH', 'LATENT'),
                        help='Student-t scale of the increments between 5 ft neighbours (ln units)')
    parser.add_argument('--anchor', type=float, nargs=2, default=(3., 1.5), metavar=('RH', 'LATENT'),
                        help='std of the weak per-cell Gaussian anchor (ln units)')
    parser.add_argument('--rml', type=int, default=0, help='Student-t RML members per point (0 = off)')
    parser.add_argument('--directional-sigma-floor', type=float, nargs=2, default=(.5, .25))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model-error', type=Path, default=None,
                        help='json table of model-error factors (build_model_error.py): the solves use the '
                             'variance Cd (1 + factor_1d^2)')
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location('compare_runs', ROOT / 'compare-runs.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)

    manifest = json.loads((args.reference.resolve() / 'run.json').read_text())
    nx, ny, nz = manifest['dims']
    dx, dy, dz = manifest['spacing_ft']
    bit = nx // 2
    data = pd.read_pickle(ROOT / manifest['data'])
    variances = pd.read_pickle(ROOT / manifest['variance'])
    tool_pairs = list(data.columns)
    sequential.dims = (nx, ny, nz)
    sequential.dX, sequential.dY, sequential.dZ = dx, dy, dz
    sequential.tools = [str(tool) for tool in tool_pairs]
    sequential.max_ratio = float(manifest['prior_ratio_max'])
    sequential.log_rv_cap = float(np.log(RV_CAP_OHMM))
    sequential.anisotropy = 'log_clipped'
    window_prior = matern_reference(manifest, 'log_clipped', .5, .5)
    gaussian = GaussianColumn(window_prior, nx, nz)
    student = StudentColumn(gaussian.mean, nz, args.dof, args.scale, args.anchor)
    truth_model = compare.Truth(manifest['benchmark'])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    indices = list(range(args.points[0], args.points[1] + 1, args.points[2]))
    if args.points_from is not None:
        history = json.loads((args.points_from / 'history.json').read_text())
        gated = {record['logging_index'] for record in history if record.get('stage') is not None}
        indices = sorted({record['logging_index'] for record in history} - gated)
        indices = [index for index in indices if args.points[0] <= index <= args.points[1]][::args.every]
        print(f'{len(indices)} points accepted as 1D by {args.points_from.name}: {indices}')

    previous_cwd = Path.cwd()
    records, cells = [], []
    try:
        os.chdir(out)
        sequential.setup_simulators(Path(manifest['benchmark']), sequential.tools)
        window = LocalWindow((nx, ny, nz), (dx, dy, dz))
        for index in range(indices[-1] + 1):
            shift, _ = window.update((sequential.WX[index], sequential.WY[index], sequential.TVD[index]))
            if index not in indices:
                continue
            started = time.monotonic()
            sequential.current_shift = shift
            observations, variance = observation_vectors(data, variances, index, tool_pairs)
            variance = directional_floor(variance, args.directional_sigma_floor)
            mask = ~noise_floor_mask(observations, variance)
            observations, variance = observations[mask], variance[mask]
            phase = phase_channels(mask, UDAR_COMPONENTS)
            target = truth_model.window({'dims': (nx, ny, nz), 'spacing_ft': (dx, dy, dz), 'shift_ft': dict(shift)})
            truth = np.concatenate([target['rh'][bit], target['ratio'][bit]]) * LN10      # ln Rh, ln(Rv/Rh)
            rng = np.random.default_rng([args.seed, index])
            clean = sequential.simulate_pred_only(truth, index, '1D')[mask]
            measured = observations
            if args.data == 'twin':
                observations = clean + np.sqrt(variance) * rng.normal(size=clean.shape)
            model_error = float(np.mean((align_phases(clean, measured, phase) - measured) ** 2 / variance))
            if args.model_error is not None:
                variance = variance * (1. + model_error_factors(args.model_error, tool_pairs)[0][mask] ** 2)

            def forward(model, reference=observations):
                return align_phases(sequential.simulate_pred_only(model, index, '1D')[mask], reference, phase)

            results = {}
            m_g, c_g, info_g = solve_map(forward, observations, variance, gaussian, gaussian.mean, nz,
                                         max_iterations=40)
            results['matern'] = (m_g, np.sqrt(np.diag(c_g)), None, info_g)
            m_t, c_t, info_t = solve_map(forward, observations, variance, student, m_g, nz, max_iterations=40)
            results['student'] = (m_t, np.sqrt(np.diag(c_t)), None, info_t)
            if args.rml:
                members = []
                for member in range(args.rml):
                    noisy = observations + np.sqrt(variance) * rng.normal(size=observations.shape)
                    centre = student.mean + rng.normal(size=student.mean.shape) / np.sqrt(student.anchor)
                    begin = clip_model(student.sample(rng), nz) if member % 2 else m_g
                    model, _, _ = solve_map(lambda m: forward(m, noisy), noisy, variance, student, begin, nz,
                                            max_iterations=30, anchor_mean=centre)
                    members.append(model)
                members = np.array(members).T
                results['student-rml'] = (members.mean(axis=1), members.std(axis=1, ddof=1), members,
                                          {'iterations': np.nan, 'objective': np.nan, 'calls': np.nan,
                                           'chi2_per_datum': np.nan})

            truth_rh = truth[:nz] / LN10
            kind = classify_column(truth_rh)
            zo = (np.arange(nz) - nz // 2) * dz
            for name, (mean, std, members, info) in results.items():
                mean_rh, std_rh = mean[:nz] / LN10, std[:nz] / LN10
                if members is None:
                    inside = np.abs(mean_rh - truth_rh) <= 1.6449 * std_rh
                else:
                    low, high = np.quantile(members[:nz] / LN10, [.05, .95], axis=1)
                    inside = (low <= truth_rh) & (truth_rh <= high)
                records.append({'index': index, 'prior': name, **info, 'model_error_chi2_per_datum': model_error})
                for cell in range(nz):
                    cells.append({'index': index, 'prior': name, 'z_ft': zo[cell], 'class': int(kind[cell]),
                                  'truth': truth_rh[cell], 'mean': mean_rh[cell], 'std': std_rh[cell],
                                  'inside': bool(inside[cell])})
            np.savez(out / f'point_{index:03d}.npz', truth=truth,
                     **{f'{name}_mean': value[0] for name, value in results.items()},
                     **{f'{name}_std': value[1] for name, value in results.items()})
            print(f'point {index}: {mask.sum()} data, {int(np.sum(np.abs(np.diff(truth_rh)) > .1))} boundaries, '
                  f'model error {model_error:.1f}; '
                  + '; '.join(f'{name} it {info["iterations"]}, chi2/n {info["chi2_per_datum"]:.2f}'
                              for name, (_, _, _, info) in results.items() if name != 'student-rml')
                  + f'; {time.monotonic() - started:.0f} s', flush=True)
    finally:
        os.chdir(previous_cwd)
    pd.DataFrame(records).to_csv(out / 'points.csv', index=False)
    table = pd.DataFrame(cells)
    table.to_csv(out / 'cells.csv', index=False)
    report(table)
    return 0


def report(table, near_ft=50.):
    table = table.assign(err=table['mean'] - table['truth'])
    table['z2'] = (table['err'] / table['std']) ** 2
    for zone, part in (('within 50 ft of the bit', table[np.abs(table['z_ft']) <= near_ft]), ('whole column', table)):
        print(f'\n{zone} (log10 Rh)')
        print('  prior        class                  cells   error  spread  z-rms  cover90')
        for name, group in part.groupby('prior', sort=False):
            rows = [('all cells', group)] + [(CLASSES[k], group[group['class'] == k]) for k in range(5)]
            for label, g in rows:
                if len(g):
                    print(f'  {name:12s} {label:21s} {len(g) / len(group):6.1%} {np.sqrt(np.mean(g["err"] ** 2)):7.3f} '
                          f'{g["std"].mean():7.3f} {np.sqrt(g["z2"].mean()):6.2f} {g["inside"].mean():8.2f}')


if __name__ == '__main__':
    raise SystemExit(main())
