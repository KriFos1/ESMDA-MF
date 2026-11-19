#!/usr/bin/env python3
"""How does the benchmark truth deviate from the Matern prior? (offline, no forward model)

The truth ln Rh is averaged onto the moving inversion window of every logging point of a run and
compared with draws of the run's Matern prior on the same window:

  marginal     values of ln Rh against the prior N(mean, std)
  increments   differences between neighbouring cells (heavy tails -> piecewise-constant earth)
  variogram    half mean squared difference against lag, along and across the window
  plausibility 2P/n = (m - mu)^T Q (m - mu) / n of the truth; 1 for a prior draw
  refit        Gaussian likelihood of the truth for other ranges (std profiled out)

python prior_truth_check.py --run runs/my-run-4 --out compare-output/prior-check
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

from matern_prior import MaternBlock

ROOT = Path(__file__).resolve().parent


def kurtosis(values):
    values = np.asarray(values, float).ravel()
    centred = values - values.mean()
    return float(np.mean(centred ** 4) / np.mean(centred ** 2) ** 2 - 3.)


def variogram(fields, axis, lags):
    """fields (count, nx, nz): half mean squared difference at each lag (cells) along axis (1 = x, 2 = z)."""
    out = []
    for lag in lags:
        a = np.take(fields, np.arange(lag, fields.shape[axis]), axis=axis)
        b = np.take(fields, np.arange(0, fields.shape[axis] - lag), axis=axis)
        out.append(.5 * np.mean((a - b) ** 2))
    return np.array(out)


def log_det_precision(block):
    lu_diag = np.abs(block.lu.U.diagonal())
    return 2. * (np.sum(np.log(lu_diag)) - np.sum(np.log(block.scale))) - block.n * np.log(block.volume)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/my-run-4')
    parser.add_argument('--out', type=Path, default=ROOT / 'compare-output/prior-check')
    parser.add_argument('--draws', type=int, default=400)
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location('compare_runs', ROOT / 'compare-runs.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    args.out.mkdir(parents=True, exist_ok=True)

    study = json.loads((args.run / 'study.json').read_text())
    prior_info = study['prior']
    truth = compare.Truth(study['benchmark'])
    indices, windows, shifts = [], [], []
    for path in sorted(args.run.glob('inversion_results_assim_*.pkl'), key=lambda p: int(p.stem.split('_')[-1])):
        with open(path, 'rb') as stream:
            saved = pickle.load(stream)
        indices.append(int(path.stem.split('_')[-1]))
        windows.append(truth.window(saved)['rh'] * compare.LN10)        # ln Rh, (nx, nz)
        shifts.append(saved['shift_ft'])
        dims, spacing = saved['dims'], saved['spacing_ft']
    indices, windows = np.array(indices), np.array(windows)
    nx, nz = dims[0], dims[2]
    dx, dz = spacing[0], spacing[2]
    n = nx * nz
    mean, std = prior_info['means'][0], prior_info['stds'][0]
    ranges = prior_info['rh_ranges_ft']
    block = MaternBlock((nx, nz), (dx, dz), ranges, mean, std)
    rng = np.random.default_rng(0)
    draws = (mean + block.color(rng.normal(size=(n, args.draws)))).T.reshape(-1, nx, nz)
    print(f'{len(indices)} windows (points {indices[0]}-{indices[-1]}), {nx} x {nz} cells of {dx:g} x {dz:g} ft; '
          f'prior ln Rh mean {mean:.3f}, std {std:.3f}, ranges {ranges} ft, nu 1')

    # --- marginal -------------------------------------------------------------------------------
    values = np.exp(windows.ravel())
    print('\nMARGINAL (ln Rh)')
    print(f'  truth: mean {windows.mean():.3f}, std {windows.std():.3f}; '
          f'{np.mean(values < 1.6):.0%} of cells at about 1.5 ohm m, {np.mean(values > 28):.0%} at about 30, '
          f'{np.mean((values >= 1.6) & (values <= 28)):.0%} in between (cells cut by a boundary)')
    for value in (1.5, 30.):
        print(f'  {value:g} ohm m sits {(np.log(value) - mean) / std:+.2f} prior std from the prior mean')

    # --- increments -----------------------------------------------------------------------------
    print('\nINCREMENTS between neighbouring cells (ln Rh)      std   |d|<0.05  |d|>1   excess kurtosis')
    increments = {}
    for name, axis in (('across window (z)', 2), ('along window (x)', 1)):
        for label, fields in (('truth', windows), ('prior', draws)):
            d = np.diff(fields, axis=axis).ravel()
            increments[name, label] = d
            print(f'  {name:18s} {label:6s} {"":14s} {d.std():6.3f} {np.mean(np.abs(d) < .05):8.1%} '
                  f'{np.mean(np.abs(d) > 1.):7.2%} {kurtosis(d):10.1f}')

    # --- variogram ------------------------------------------------------------------------------
    lags_z, lags_x = np.arange(1, nz // 2), np.arange(1, nx - 4)
    vario = {('z', 'truth'): variogram(windows, 2, lags_z), ('z', 'prior'): variogram(draws, 2, lags_z),
             ('x', 'truth'): variogram(windows, 1, lags_x), ('x', 'prior'): variogram(draws, 1, lags_x)}
    print('\nVARIOGRAM (half mean squared difference of ln Rh)')
    for direction, lags, step in (('z', lags_z, dz), ('x', lags_x, dx)):
        print(f'  lag {direction} (ft)   ' + ' '.join(f'{lag * step:6.0f}' for lag in lags[[0, 1, 3, 7, 15, -1]]))
        for label in ('truth', 'prior'):
            print(f'    {label:10s}   ' + ' '.join(f'{v:6.3f}' for v in vario[direction, label][[0, 1, 3, 7, 15, -1]]))

    # --- plausibility ---------------------------------------------------------------------------
    def two_p_over_n(blk, fields):
        white = blk.whiten(fields.reshape(len(fields), -1).T - blk.mean[:, None])
        return np.sum(white ** 2, axis=0) / blk.n, white

    score_truth, white_truth = two_p_over_n(block, windows)
    score_draws, white_draws = two_p_over_n(block, draws)
    top = np.sort(white_truth ** 2, axis=0)[::-1]
    share = top[:n // 20].sum(axis=0) / top.sum(axis=0)
    print('\nPLAUSIBILITY 2P/n (1 for a draw of the prior)')
    print(f'  prior draws: mean {score_draws.mean():.2f}, 1-99 % {np.quantile(score_draws, .01):.2f}-'
          f'{np.quantile(score_draws, .99):.2f}')
    print(f'  truth windows: median {np.median(score_truth):.2f}, 10-90 % {np.quantile(score_truth, .1):.2f}-'
          f'{np.quantile(score_truth, .9):.2f}, max {score_truth.max():.1f}; '
          f'{np.mean(score_truth > np.quantile(score_draws, .99)):.0%} of windows above the prior 99 % level, '
          f'{np.mean(score_truth < np.quantile(score_draws, .01)):.0%} below the 1 % level')
    print(f'  whitened truth: excess kurtosis {kurtosis(white_truth):.0f} (prior draws {kurtosis(white_draws):.1f}); '
          f'the 5 % largest cells carry {np.median(share):.0%} of the penalty (prior draws '
          f'{np.median(np.sort(white_draws ** 2, axis=0)[::-1][:n // 20].sum(axis=0) / (white_draws ** 2).sum(axis=0)):.0%})')

    # --- refit ----------------------------------------------------------------------------------
    subset = windows[::10]                      # windows 10 points apart overlap little along the well
    print(f'\nREFIT on {len(subset)} windows: Gaussian log-likelihood per cell relative to the current prior, '
          f'std profiled out (best std in brackets)')
    grid_x, grid_z = (25., 50., 100., 200., 400.), (5., 10., 20., 35., 50., 100.)
    table = np.zeros((len(grid_x), len(grid_z), 2))
    for i, rx in enumerate(grid_x):
        for j, rz in enumerate(grid_z):
            blk = MaternBlock((nx, nz), (dx, dz), (rx, rz), subset.mean(), 1.)
            s2 = two_p_over_n(blk, subset)[0].mean()
            table[i, j] = (-.5 - .5 * np.log(s2) + .5 * log_det_precision(blk) / n, np.sqrt(s2))
    current = MaternBlock((nx, nz), (dx, dz), ranges, mean, std)
    reference = -.5 * two_p_over_n(current, subset)[0].mean() + .5 * log_det_precision(current) / n
    print('  range x \\ z  ' + ' '.join(f'{rz:12g}' for rz in grid_z))
    for i, rx in enumerate(grid_x):
        print(f'  {rx:9g}    ' + ' '.join(f'{table[i, j, 0] - reference:+6.2f} ({table[i, j, 1]:4.2f})'
                                          for j in range(len(grid_z))))
    best = np.unravel_index(np.argmax(table[..., 0]), table.shape[:2])
    best_block = MaternBlock((nx, nz), (dx, dz), (grid_x[best[0]], grid_z[best[1]]), subset.mean(), table[best][1])
    score_best, white_best = two_p_over_n(best_block, windows)
    print(f'  best: range x {grid_x[best[0]]:g} ft, z {grid_z[best[1]]:g} ft, std {table[best][1]:.2f}, mean '
          f'{subset.mean():.2f}; whitened truth under it still has excess kurtosis {kurtosis(white_best):.0f}')

    # --- figure ---------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, c_truth, c_prior = '#1f2933', '#6b7785', '#b4532a', '#2f6f9f'
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': muted, 'axes.labelcolor': ink, 'xtick.color': muted, 'ytick.color': muted})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    example = int(np.argmin(np.abs(indices - 115)))
    xo, zo = (np.arange(nx) - nx // 2) * dx, (np.arange(nz) - nz // 2) * dz
    limits = dict(vmin=np.log(1.), vmax=np.log(40.), cmap='viridis', shading='auto')
    for ax, field, title in ((axes[0, 0], windows[example], f'Truth on the window, point {indices[example]}'),
                             (axes[0, 1], draws[0], 'One draw of the Matern prior')):
        mesh = ax.pcolormesh(xo, zo, field.T, **limits)
        ax.invert_yaxis()
        ax.set(title=title, xlabel='along well from bit (ft)', ylabel='depth from bit (ft)')
    bar = fig.colorbar(mesh, ax=axes[0, :2], shrink=.8, ticks=np.log([1, 2, 5, 10, 20, 40]))
    bar.ax.set_yticklabels(['1', '2', '5', '10', '20', '40'])
    bar.set_label('Rh (ohm m)')

    ax = axes[0, 2]
    bins = np.linspace(-3.5, 3.5, 71)
    for label, colour in (('truth', c_truth), ('prior', c_prior)):
        ax.hist(increments['across window (z)', label], bins=bins, density=True, histtype='step', lw=2,
                color=colour, label=label)
    ax.set(yscale='log', title='Difference between vertical neighbours', xlabel='difference in ln Rh',
           ylabel='density (log scale)')
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for direction, lags, step, style in (('z', lags_z, dz, '-'), ('x', lags_x, dx, '--')):
        for label, colour in (('truth', c_truth), ('prior', c_prior)):
            ax.plot(lags * step, vario[direction, label], style, lw=2, color=colour,
                    label=f'{label}, {"across (z)" if direction == "z" else "along (x)"}')
    ax.set(title='Variogram', xlabel='lag (ft)', ylabel='half mean squared difference', ylim=(0, None))
    ax.legend(frameon=False)

    ax = axes[1, 1]
    lo, hi = np.quantile(score_draws, [.01, .99])
    ax.axhspan(lo, hi, color=c_prior, alpha=.25, lw=0, label='prior draws, 1-99 %')
    ax.plot(indices, score_truth, lw=2, color=c_truth, label='truth')
    ax.set(yscale='log', title='Plausibility of the truth under the prior', xlabel='logging point',
           ylabel='2P/n (1 = typical prior draw)')
    ax.legend(frameon=False)

    ax = axes[1, 2]
    bins = np.linspace(-12, 12, 97)
    ax.hist(white_truth.ravel(), bins=bins, density=True, histtype='step', lw=2, color=c_truth, label='truth')
    ax.hist(white_draws.ravel(), bins=bins, density=True, histtype='step', lw=2, color=c_prior, label='prior')
    ax.set(yscale='log', title='Whitened field', xlabel='whitened value (standard normal under the prior)',
           ylabel='density (log scale)')
    ax.legend(frameon=False)
    fig.savefig(args.out / 'prior_truth_check.png', dpi=150)
    np.savez(args.out / 'prior_truth_check.npz', indices=indices, two_p_over_n=score_truth)
    print(f'\nFigure: {args.out / "prior_truth_check.png"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
