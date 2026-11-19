#!/usr/bin/env python3
"""
Compare the memory (rho) settings of the fault-study runs and show how the knobs work.

The earlier version of this script (now rank_runs.py) ranked runs by `posterior_misfit` at one logging
point. That number is a data misfit: a run that forgets everything fits the data best and still knows
least about the formation, so it cannot rank memory settings. This script scores every run against the
benchmark truth instead, at every logging point, for both the carried prior (what the memory setting
hands to the next point) and the final posterior.

Per logging point and zone (near the bit, ahead of it, whole window), on log10 Rh:
  rmse     error of the ensemble mean
  spread   ensemble standard deviation
  z_rms    rms of (mean - truth) / std ; 1 = calibrated, > 1 = overconfident, < 1 = too wide
  cover90  fraction of cells whose truth lies inside the 5-95 % ensemble band (target 0.90)
  crps     continuous ranked probability score (Gaussian); one number for accuracy AND calibration,
           lower is better. This is the ranking score.
The same scores are computed for Rv and for the ratio Rv/Rh (columns *_rv_* and *_ratio_*; their CRPS is
taken from the members because the clipped ratio is not Gaussian, and z_rms is omitted).
Plus the data-side numbers: forecast chi2/n of the carried prior, gate openings, 2D MAP iterations/time.

Outputs (in --out, default compare-output/):
  points.csv        one row per run and logging point
  summary.csv       one row per run, ranked by posterior near-bit CRPS
  knob_effects.csv  every pair of runs that differ in exactly one setting, with the paired difference
  fig1_memory_knobs.png   how each knob changes what is kept per step (pure formula, no run data)
  fig2_scoreboard.png     all runs, ranked
  fig3_mean_memory.png    scores against the memory length of the ensemble mean (memory_ft * mean_factor)
  fig4_tradeoff.png       accuracy against calibration
  fig5_along_well.png     base run and the best run of each memory length, point by point
  fig6_profiles.png       error and spread across the window, behind / at / ahead of the bit
  fig7_anisotropy.png     Rv and Rv/Rh scores (empirical CRPS), the test of --memory-latent-factor

Usage
-----
  python compare-runs.py runs                       # fault* runs, points 101-140
  python compare-runs.py runs --glob '*' --refresh  # every run folder, recompute the cache
"""
import argparse
import json
import pickle
import os
import sys
from math import erf, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

LN10 = np.log(10.)
NEAR_X_FT, FAR_X_FT, ZONE_Z_FT = 25., 80., 50.     # zone limits relative to the bit
SETTINGS = ['memory_ft', 'mean_factor', 'ahead_factor', 'surprise_factor', 'surprise_trigger', 'surprise_low',
            'surprise_high',
            'taper_ft', 'latent_factor', 'gate', 'seed', 'analysis', 'mda_steps', 'ensemble_size',
            'likelihood_weight', 'sigma_floor', 'twin']
DEFAULTS = {'mean_factor': 1., 'ahead_factor': 1., 'surprise_factor': 1., 'surprise_trigger': 'none',
            'surprise_low': 0., 'surprise_high': 0.,
            'taper_ft': 40., 'latent_factor': 1.}
ZONES = ['near', 'ahead', 'window']
PHI = np.vectorize(lambda v: .5 * (1. + erf(v / sqrt(2.))))

# Chart colours (validated categorical slots 1-3, neutral inks)
C_MEM = {10.: '#eb6834', 50.: '#2a78d6', 100.: '#1baf7a'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


# ----------------------------------------------------------------------
# Truth
# ----------------------------------------------------------------------
def overlap_weights(centers, edges, width):
    lo, hi = centers - width / 2, centers + width / 2
    w = np.maximum(0., np.minimum(edges[1:, None], hi) - np.maximum(edges[:-1, None], lo)) / width
    for i in np.flatnonzero(w.sum(axis=1) == 0):
        w[i, np.argmin(np.abs(centers - (edges[i] + edges[i + 1]) / 2))] = 1.
    return w / w.sum(axis=1, keepdims=True)


class Truth:
    """Benchmark section upscaled onto the moving inversion window.

    ``upscaling`` 'layered' (default): Rh is the harmonic and Rv the arithmetic average of the benchmark cells
    in a window cell, the effective medium of thin beds (multistart_point.benchmark_window). This is the model
    on the grid that reproduces the data: the 2D misfit of the truth on the 5 ft grid is 3-9 per datum, against
    119-483 for 'arithmetic' (both averaged arithmetically, refine_z_check.py), which was used before
    2026-11-18 and penalises a correct estimate in every cell that holds a bed boundary.
    The environment variable TRUTH_UPSCALING overrides the default (for comparisons with older scores).
    """

    def __init__(self, benchmark, upscaling=None):
        self.upscaling = upscaling or os.environ.get('TRUTH_UPSCALING', 'layered')
        if self.upscaling not in ('layered', 'arithmetic'):
            raise ValueError(f'Unknown truth upscaling {self.upscaling!r}')
        if Path(benchmark).suffix == '.npz':           # synthetic truth of a twin experiment (--twin-seed)
            with np.load(benchmark) as f:
                self.x, self.z, self.rh, self.rv = f['x'], f['z'], f['rh'], f['rv']
            return
        import h5py
        with h5py.File(Path(benchmark) / 'globalmodel.h5', 'r') as f:
            self.x = f['formation/x'][:, 0] / .3048
            self.z = f['formation/z'][:, 0] / .3048
            self.rh = f['formation/Rh'][:, f['formation/Rh'].shape[1] // 2, :]
            self.rv = f['formation/Rv'][:, f['formation/Rv'].shape[1] // 2, :]

    def window(self, saved):
        nx, _, nz = saved['dims']
        dx, _, dz = saved['spacing_ft']
        s = saved['shift_ft']
        wx = overlap_weights(self.x, s['x'] + np.arange(nx + 1) * dx, float(np.median(np.diff(self.x))))
        wz = overlap_weights(self.z, s['z'] + np.arange(nz + 1) * dz, float(np.median(np.diff(self.z))))
        rv = np.log(wx @ self.rv.T @ wz.T) / LN10
        if self.upscaling == 'layered':
            rh = -np.log(wx @ (1. / self.rh.T) @ wz.T) / LN10
        else:
            rh = np.log(wx @ self.rh.T @ wz.T) / LN10
        return {'rh': rh, 'rv': rv, 'ratio': rv - rh}


def anisotropy_fields(state, saved, cells, shape):
    """ln Rv and ln(Rv/Rh) ensembles from a (2 * cells, members) state, as the forward model sees them."""
    log_rh, latent = state[:cells], state[cells:2 * cells]
    kind = saved.get('anisotropy', 'log_clipped')
    lo, hi = saved.get('rv_ratio_bounds', (1., 3.))
    if kind == 'log_clipped':
        ratio = np.maximum(latent, 0.)
    elif kind == 'probit':
        ratio = np.log(lo) + np.log(hi / lo) * PHI(latent)
    else:
        raise ValueError(f'Unknown anisotropy parameterization {kind!r}')
    log_rv = log_rh + ratio
    cap = saved.get('rv_hard_cap_ohmm')
    if cap:
        log_rv = np.minimum(log_rv, np.log(cap))
    return log_rv.reshape(*shape, -1), (log_rv - log_rh).reshape(*shape, -1)


# ----------------------------------------------------------------------
# Reading a run
# ----------------------------------------------------------------------
def run_settings(folder):
    study = json.loads((folder / 'study.json').read_text())
    relax = study.get('relaxation')
    gate, analysis, twin = study.get('gate'), study.get('analysis'), study.get('twin')
    out = {'run': folder.name, 'start': study.get('start'), 'seed': study.get('seed'),
           'gate': gate['threshold'] if isinstance(gate, dict) else np.inf,
           'analysis': analysis['method'] if isinstance(analysis, dict) else str(analysis or 'esmda'),
           'mda_steps': study.get('mda_steps'), 'ensemble_size': study.get('ensemble_size'),
           'likelihood_weight': study.get('likelihood_weight', 1.),
           'sigma_floor': '/'.join(f'{v:g}' for v in study.get('directional_sigma_floor') or []) or 'none',
           'twin': twin['seed'] if isinstance(twin, dict) else -1,
           'benchmark': str(folder / 'twin_truth.npz') if isinstance(twin, dict) else study.get('benchmark'),
           'memory_ft': np.inf, **DEFAULTS}
    if isinstance(relax, dict):
        out['memory_ft'] = relax['memory_ft']
        adaptive = relax.get('adaptive') or {}
        for key in ('mean_factor', 'ahead_factor', 'surprise_factor', 'taper_ft', 'latent_factor'):
            out[key] = adaptive.get(key, DEFAULTS[key])
        if out['surprise_factor'] != 1:
            # Runs before the ahead factor / taper were removed triggered the surprise on the previous
            # point's post-fit misfit (which is ~0 after a 2D point); later runs use the forecast misfit.
            out['surprise_trigger'] = adaptive.get('surprise_trigger', 'post_fit')
            out['surprise_low'], out['surprise_high'] = adaptive.get('surprise_misfit', [3., 25.])
    # The taper does nothing when both factors it acts through are off.
    if out['ahead_factor'] == 1 and out['surprise_factor'] == 1:
        out['taper_ft'] = DEFAULTS['taper_ft']
    return out


def field_scores(fields, truth, zones, empirical=False):
    """fields: (nx, nz, members) natural-log field. Returns {zone_metric: value} and the per-column
    error/spread. ``empirical`` scores the CRPS from the members themselves (for the anisotropy fields,
    which are clipped and far from Gaussian; z_rms is then not meaningful and left out)."""
    f = fields / LN10
    mean, std = f.mean(axis=2), f.std(axis=2, ddof=1)
    err = mean - truth
    inside = (np.quantile(f, .05, axis=2) <= truth) & (truth <= np.quantile(f, .95, axis=2))
    if empirical:
        n = f.shape[2]
        ordered = np.sort(f, axis=2)
        pair = 2. * (ordered * (2. * np.arange(1, n + 1) - n - 1.)).sum(axis=2) / n ** 2   # E|X - X'|
        crps = np.abs(f - truth[:, :, None]).mean(axis=2) - .5 * pair
        z = np.full_like(err, np.nan)
    else:
        z = err / std
        crps = std * (z * (2. * PHI(z) - 1.) + 2. * np.exp(-.5 * z ** 2) / sqrt(2. * pi) - 1. / sqrt(pi))
    out = {}
    for name in ZONES:
        mask = zones[name]
        out[f'rmse_{name}'] = float(np.sqrt(np.mean(err[mask] ** 2)))
        out[f'spread_{name}'] = float(np.mean(std[mask]))
        out[f'z_rms_{name}'] = float(np.sqrt(np.mean(z[mask] ** 2)))
        out[f'cover90_{name}'] = float(np.mean(inside[mask]))
        out[f'crps_{name}'] = float(np.mean(crps[mask]))
    rows = zones['_rows']
    return out, np.sqrt(np.mean(err[:, rows] ** 2, axis=1)), std[:, rows].mean(axis=1)


def collect_run(folder, truth, first, last):
    """Per-point table and per-column profiles of one run (logging points first..last)."""
    history = json.loads((folder / 'history.json').read_text())
    one_d = {r['logging_index']: r for r in history if r.get('stage') is None}
    two_d = {r['logging_index']: r for r in history if r.get('stage') is not None}
    rows, profiles = [], {}
    for index in range(first, last + 1):
        path = folder / f'inversion_results_assim_{index}.pkl'
        if not path.exists() or index not in one_d:
            continue
        with path.open('rb') as stream:
            saved = pickle.load(stream)
        nx, _, nz = saved['dims']
        dx, _, dz = saved['spacing_ft']
        sign = 1 if saved['window_movement'][0] >= 0 else -1
        xo = sign * (np.arange(nx) - nx // 2) * dx                       # ft ahead of the bit
        zo = (np.arange(nz) - nz // 2) * dz
        rows_z = np.abs(zo) <= ZONE_Z_FT
        zones = {'near': np.outer(np.abs(xo) <= NEAR_X_FT, rows_z),
                 'ahead': np.outer((xo > NEAR_X_FT) & (xo <= FAR_X_FT), rows_z),
                 'window': np.ones((nx, nz), bool), '_rows': rows_z}
        target = truth.window(saved)
        prior_state = np.asarray(saved['prior_params'])
        post_state = np.asarray(saved['posterior_ensemble_params']).T
        prior = prior_state[:nx * nz].reshape(nx, nz, -1)
        post = post_state[:nx * nz].reshape(nx, nz, -1)
        p_scores, p_err, p_std = field_scores(prior, target['rh'], zones)
        q_scores, q_err, q_std = field_scores(post, target['rh'], zones)
        # Anisotropy: Rv and the ratio Rv/Rh (what --memory-latent-factor acts on)
        for stage, state, scores in (('prior', prior_state, p_scores), ('post', post_state, q_scores)):
            fields = dict(zip(('rv', 'ratio'), anisotropy_fields(state, saved, nx * nz, (nx, nz))))
            for name, field in fields.items():
                extra, _, _ = field_scores(field, target[name], zones, empirical=True)
                scores.update({f'{name}_{k}': v for k, v in extra.items() if not k.startswith('z_rms')})
        record, stage = one_d[index], two_d.get(index, {})
        gate = record.get('gate') or {}
        row = {'run': folder.name, 'index': index,
               'stage': {'laplace_2d': '2D', 'carried_prior_fallback': 'fallback'}.get(stage.get('stage'), '1D'),
               'rho': record.get('relaxation_rho'), 'surprise': record.get('relaxation_surprise'),
               'forecast_chi2': gate.get('pre_update_mean_prediction_misfit',
                                         record['prior']['mean_prediction_1d_misfit']),
               'post1d_chi2': gate.get('post_member_median_misfit',
                                       record['posterior']['member_1d_misfit_median']),
               'gate_open': bool(gate.get('open', False)),
               'map_iterations': (stage.get('map') or {}).get('accepted_iterations', np.nan),
               'map_seconds': (stage.get('map') or {}).get('seconds', np.nan),
               'map_misfit': (stage.get('map') or {}).get('misfit', np.nan)}
        row.update({f'prior_{k}': v for k, v in p_scores.items()})
        row.update({f'post_{k}': v for k, v in q_scores.items()})
        rows.append(row)
        profiles[index] = np.stack([xo, p_err, p_std, q_err, q_std])
    return pd.DataFrame(rows), profiles


def load_all(root, pattern, out, first, last, refresh):
    cache = out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    settings, points, profiles, truths = [], [], {}, {}
    folders = sorted(d for d in root.glob(pattern) if (d / 'history.json').exists() and (d / 'study.json').exists())
    if not folders:
        sys.exit(f"No run folders with history.json and study.json match '{pattern}' under {root}")
    for folder in folders:
        info = run_settings(folder)
        settings.append(info)
        table, prof = cache / f'{folder.name}.csv', cache / f'{folder.name}.npz'
        stale = refresh or not table.exists() or table.stat().st_mtime < (folder / 'history.json').stat().st_mtime
        if not stale and 'post_ratio_crps_near' not in table.open().readline():
            stale = True                                   # cache written before the anisotropy scores
        upscaling = os.environ.get('TRUTH_UPSCALING', 'layered')
        if not stale:
            cached = pd.read_csv(table, nrows=1)
            stale = 'truth_upscaling' not in cached or cached['truth_upscaling'].iloc[0] != upscaling
        if stale:
            print(f'  scoring {folder.name} ...', flush=True)
            truth = truths.setdefault(info['benchmark'], Truth(info['benchmark']))
            frame, p = collect_run(folder, truth, first, last)
            frame['truth_upscaling'] = truth.upscaling
            frame.to_csv(table, index=False)
            np.savez(prof, **{str(k): v for k, v in p.items()})
        frame = pd.read_csv(table)
        frame = frame[(frame['index'] >= first) & (frame['index'] <= last)]
        points.append(frame)
        with np.load(prof) as data:
            profiles[folder.name] = {int(k): data[k] for k in data.files if first <= int(k) <= last}
    return pd.DataFrame(settings).set_index('run'), pd.concat(points, ignore_index=True), profiles


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def label(s):
    """Short, readable description of a run's memory setting."""
    if not np.isfinite(s['memory_ft']):
        return 'no relaxation'
    bits = [f"mem {s['memory_ft']:g} ft"]
    if s['mean_factor'] != 1:
        bits.append(f"mean x{s['mean_factor']:g}")
    if s['surprise_trigger'] == 'forecast':
        bits.append(f"surprise {s['surprise_factor']:g} [forecast {s['surprise_low']:g}-{s['surprise_high']:g}]")
    elif s['ahead_factor'] != 1 or s['surprise_factor'] != 1:
        bits.append(f"ahead {s['ahead_factor']:g}/surp {s['surprise_factor']:g}")
        if s['taper_ft'] != 40:
            bits.append(f"taper {s['taper_ft']:g}")
        if s['surprise_high'] != 25:
            bits.append(f"surp-hi {s['surprise_high']:g}")
    if s.get('seed', 2031) != 2031:
        bits.append(f"seed {s['seed']:g}")
    if s['latent_factor'] != 1:
        bits.append(f"latent x{s['latent_factor']:g}")
    if s['gate'] != 25:
        bits.append(f"gate {s['gate']:g}" if np.isfinite(s['gate']) else 'no gate')
    if s['analysis'] != 'enif':
        bits.append(s['analysis'])
    if s['mda_steps'] != 10:
        bits.append(f"{s['mda_steps']:g} steps")
    if s['ensemble_size'] != 250:
        bits.append(f"{s['ensemble_size']:g} members")
    if s['likelihood_weight'] != 1:
        bits.append(f"likelihood x{s['likelihood_weight']:g}")
    if s['sigma_floor'] != '0.5/0.25':
        bits.append(f"sigma floor {s['sigma_floor']}")
    if s['twin'] >= 0:
        bits.append(f"TWIN {s['twin']:g}")
    return ', '.join(bits)


SUMMARY_COLUMNS = ['post_crps_near', 'post_rmse_near', 'post_z_rms_near', 'post_cover90_near',
                   'post_crps_ahead', 'post_rmse_ahead', 'post_z_rms_ahead', 'post_rmse_window',
                   'prior_crps_near', 'prior_rmse_near', 'prior_z_rms_near', 'prior_crps_ahead',
                   'prior_rmse_ahead', 'prior_z_rms_ahead', 'post_spread_near',
                   'post_rv_crps_near', 'post_rv_rmse_near', 'post_rv_cover90_near',
                   'post_ratio_crps_near', 'post_ratio_rmse_near', 'post_ratio_cover90_near',
                   'post_ratio_spread_near', 'prior_ratio_crps_near', 'prior_ratio_rmse_near',
                   'prior_ratio_crps_ahead', 'prior_rv_crps_near']


def summarise(settings, points):
    grouped = points.groupby('run')
    summary = grouped[SUMMARY_COLUMNS].mean()
    summary['forecast_chi2_gmean'] = grouped['forecast_chi2'].apply(lambda v: float(np.exp(np.log(v).mean())))
    summary['points'] = grouped['index'].count()
    summary['points_2d'] = grouped['gate_open'].sum()
    summary['map_iterations'] = grouped['map_iterations'].mean()
    summary['map_hours'] = grouped['map_seconds'].sum() / 3600.
    summary['map_misfit'] = grouped['map_misfit'].mean()
    summary = settings.drop(columns='benchmark').join(summary, how='inner')
    summary.insert(1, 'mean_memory_ft', summary['memory_ft'] * summary['mean_factor'])
    summary.insert(0, 'setting', [label(s) for _, s in summary.iterrows()])
    return summary.sort_values('post_crps_near')


def paired(points, a, b, column):
    """Mean of (b - a) over the common logging points and a standard error that allows for the
    point-to-point correlation along the well (lag-1 effective sample size)."""
    pa = points[points['run'] == a].set_index('index')[column]
    pb = points[points['run'] == b].set_index('index')[column]
    d = (pb - pa).dropna().to_numpy()
    if len(d) < 3:
        return np.nan, np.nan
    r = np.corrcoef(d[:-1], d[1:])[0, 1] if d.std() > 0 else 0.
    n_eff = max(2., len(d) * (1. - r) / (1. + r)) if np.isfinite(r) and r > 0 else len(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(n_eff))


def knob_effects(settings, points):
    rows = []
    names = list(settings.index)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if settings.loc[a, 'start'] != settings.loc[b, 'start']:
                continue
            diff = [k for k in SETTINGS if settings.loc[a, k] != settings.loc[b, k]]
            # ahead + surprise factors were always changed together: treat them as one knob
            if set(diff) == {'ahead_factor', 'surprise_factor'}:
                diff = ['ahead_factor']
            # switching the surprise on changes its factor, trigger and thresholds together: one knob
            surprise_keys = {'surprise_factor', 'surprise_trigger', 'surprise_low', 'surprise_high'}
            if 'surprise_factor' in diff and set(diff) <= surprise_keys and 'none' in (
                    settings.loc[a, 'surprise_trigger'], settings.loc[b, 'surprise_trigger']):
                diff = ['surprise_factor']
            if len(diff) != 1:
                continue
            knob = diff[0]
            lo, hi = sorted([a, b], key=lambda n: settings.loc[n, knob])
            # "from" is the plainer setting: factor nearer 1 / shorter memory / default value
            if knob in ('ahead_factor', 'surprise_factor'):
                lo, hi = hi, lo
            row = {'knob': 'ahead+surprise' if knob == 'ahead_factor' else knob,
                   'from': settings.loc[lo, knob], 'to': settings.loc[hi, knob],
                   'context': label(settings.loc[lo]), 'run_from': lo, 'run_to': hi}
            for column in ('post_crps_near', 'post_rmse_near', 'post_z_rms_near', 'prior_crps_near',
                           'prior_crps_ahead', 'post_rv_crps_near', 'post_ratio_crps_near',
                           'post_ratio_rmse_near', 'prior_ratio_crps_near', 'prior_ratio_crps_ahead'):
                mean, se = paired(points, lo, hi, column)
                base = points[points['run'] == lo][column].mean()
                row[f'd_{column}_pct'] = 100. * mean / base
                row[f'd_{column}_se_pct'] = 100. * se / base
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=['knob', 'from', 'to', 'context', 'run_from', 'run_to'])
    return pd.DataFrame(rows).sort_values(['knob', 'd_post_crps_near_pct']).reset_index(drop=True)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def style(plt):
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'text.color': INK, 'xtick.color': INK2,
        'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'lines.linewidth': 2,
        'legend.frameon': False, 'figure.dpi': 130})


def fig_knobs(plt, out, step=5.):
    """How the knobs act: everything reduces to 'how much is kept per step', per window column."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), constrained_layout=True)
    k = np.arange(0, 41)

    ax = axes[0]
    for mem in (10., 50., 100.):
        ax.plot(k * step, np.exp(-2. * k * step / mem), color=C_MEM[mem])
        x_lab = {10.: 9., 50.: 52., 100.: 120.}[mem]
        ax.text(x_lab, np.exp(-2. * x_lab / mem) + .04, f'memory {mem:g} ft', color=INK, fontsize=8.5)
    ax.set(title='1  --memory-ft: how fast the ensemble forgets',
           xlabel='distance drilled since the information arrived (ft); points are 5 ft apart',
           ylabel='share of the variance reduction still kept', ylim=(0, 1.02), xlim=(0, 200))

    ax = axes[1]
    for factor, dash in ((1., '-'), (2., (0, (5, 2))), (4., (0, (2, 1.5))), (8., (0, (1, 1)))):
        ax.plot(k * step, np.exp(-k * step / (50. * factor)), color=C_MEM[50.], linestyle=dash)
        ax.text(203, np.exp(-200. / (50. * factor)), f'x{factor:g}', va='center', fontsize=8.5, clip_on=False)
    ax.plot(k * step, np.exp(-k * step / 40.), color=C_MEM[10.], linestyle=(0, (2, 1.5)))
    ax.text(40, .15, 'memory 10 ft, x4', fontsize=8.5)
    ax.text(110, .93, 'memory 50 ft, mean factor:', fontsize=8.5)
    ax.set(title='2  --memory-mean-factor: the mean outlives the spread',
           xlabel='distance drilled since the information arrived (ft)',
           ylabel="share of the mean's departure from the prior kept", ylim=(0, 1.02), xlim=(0, 200))

    fig.savefig(out / 'fig1_memory_knobs.png')
    plt.close(fig)


def fig_scoreboard(plt, out, summary):
    panels = [('post_crps_near', 'Posterior CRPS near bit\n(accuracy + calibration)'),
              ('post_rmse_near', 'Posterior Rh RMSE near bit\n(log10)'),
              ('post_z_rms_near', 'Calibration z-rms near bit\n(1 = honest, >1 too narrow)'),
              ('prior_crps_ahead', 'Carried-prior CRPS ahead\n(look-ahead value)'),
              ('map_misfit', '2D MAP data misfit\n(old ranking metric)')]
    s = summary.iloc[::-1]
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, len(panels), figsize=(15, .33 * len(s) + 1.9), sharey=True,
                             constrained_layout=True)
    colours = [C_MEM.get(m, INK2) for m in s['memory_ft']]
    for ax, (column, title) in zip(axes, panels):
        ax.hlines(y, s[column].min(), s[column], color=GRID, linewidth=1)
        ax.scatter(s[column], y, c=colours, s=42, zorder=3, edgecolor=SURFACE, linewidth=1)
        if column == 'post_z_rms_near':
            ax.axvline(1., color=INK2, linewidth=.8, linestyle=(0, (3, 2)))
        ax.set_title(title, fontsize=9, fontweight='normal')
        ax.grid(axis='y', visible=False)
        ax.margins(x=.08)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(s['setting'], fontsize=8.5)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=f'memory {m:g} ft')
               for m, c in C_MEM.items() if m in set(summary['memory_ft'])]
    fig.legend(handles=handles, loc='outside upper right', ncols=3)
    fig.suptitle('Runs ranked by posterior CRPS near the bit (best on top; lower is better in every column)', x=.01, ha='left',
                 fontweight='bold', fontsize=11)
    fig.savefig(out / 'fig2_scoreboard.png')
    plt.close(fig)


def fig_tradeoff(plt, out, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for ax, stage, title in ((axes[0], 'prior', 'Carried prior (what the memory hands on)'),
                             (axes[1], 'post', 'Final posterior')):
        for mem, group in summary.groupby('memory_ft'):
            ax.scatter(group[f'{stage}_z_rms_near'], group[f'{stage}_rmse_near'], s=48, zorder=3,
                       color=C_MEM.get(mem, INK2), edgecolor=SURFACE, linewidth=1, label=f'memory {mem:g} ft')
        ax.axvline(1., color=INK2, linewidth=.8, linestyle=(0, (3, 2)))
        low = summary[f'{stage}_rmse_near'].min()
        ax.annotate('calibrated', (1., low), xytext=(4, -2), textcoords='offset points', fontsize=8,
                    color=INK2, va='top')
        picks = {summary[f'{stage}_crps_near'].idxmin(): 'best CRPS'}
        for name in summary.index:
            if summary.loc[name, 'setting'] == 'mem 50 ft':
                picks[name] = 'base (mem 50 only)'
        for name, text in picks.items():
            ax.annotate(f"{text}:\n{summary.loc[name, 'setting']}",
                        (summary.loc[name, f'{stage}_z_rms_near'], summary.loc[name, f'{stage}_rmse_near']),
                        xytext=(8, 8) if text.startswith('base') else (-175, 28),
                        textcoords='offset points', fontsize=8,
                        arrowprops=dict(arrowstyle='-', color=INK2, linewidth=.6))
        ax.set(title=title, xlabel='calibration z-rms near bit  (right = overconfident)',
               ylabel='Rh RMSE near bit (log10)')
    axes[0].legend(loc='upper left')
    fig.suptitle('Accuracy against honesty: longer memory lowers the error but narrows the spread too far',
                 x=.01, ha='left', fontweight='bold', fontsize=11)
    fig.savefig(out / 'fig4_tradeoff.png')
    plt.close(fig)


def fig_mean_memory(plt, out, summary):
    """The scores collapse onto one curve when plotted against the memory length of the ensemble mean."""
    panels = [('post_crps_near', 'Posterior CRPS near bit (lower = better)'),
              ('prior_crps_ahead', 'Carried-prior CRPS ahead of bit (lower = better)'),
              ('post_z_rms_near', 'Posterior calibration z-rms near bit (1 = honest)')]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    for ax, (column, title) in zip(axes, panels):
        for mem, group in summary.groupby('memory_ft'):
            ax.scatter(group['mean_memory_ft'], group[column], s=48, zorder=3, color=C_MEM.get(mem, INK2),
                       edgecolor=SURFACE, linewidth=1, label=f'spread memory {mem:g} ft')
        ax.set_xscale('log')
        ticks = sorted(set(summary['mean_memory_ft']))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:g}' for t in ticks])
        ax.minorticks_off()
        ax.set(title=title, xlabel='mean memory = memory_ft x mean_factor (ft)')
    axes[0].legend(loc='upper right')
    fig.suptitle('One number explains most of the ranking: how long the ensemble MEAN is remembered',
                 x=.01, ha='left', fontweight='bold', fontsize=11)
    fig.savefig(out / 'fig3_mean_memory.png')
    plt.close(fig)


def fig_anisotropy(plt, out, summary):
    """Rv and Rv/Rh scores: the only place --memory-latent-factor could show up."""
    panels = [('post_ratio_crps_near', 'Posterior Rv/Rh CRPS near bit'),
              ('post_ratio_rmse_near', 'Posterior Rv/Rh RMSE near bit\n(log10)'),
              ('prior_ratio_crps_near', 'Carried-prior Rv/Rh CRPS near bit'),
              ('post_rv_crps_near', 'Posterior Rv CRPS near bit'),
              ('post_crps_near', 'Posterior Rh CRPS near bit\n(the main ranking score)')]
    s = summary.sort_values('post_ratio_crps_near').iloc[::-1]
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, len(panels), figsize=(15, .33 * len(s) + 1.9), sharey=True,
                             constrained_layout=True)
    latent = (s['latent_factor'] != 1).to_numpy()
    colours = np.array([C_MEM.get(m, INK2) for m in s['memory_ft']])
    for ax, (column, title) in zip(axes, panels):
        ax.hlines(y, s[column].min(), s[column], color=GRID, linewidth=1)
        ax.scatter(s[column][~latent], y[~latent], c=list(colours[~latent]), s=42, zorder=3,
                   edgecolor=SURFACE, linewidth=1)
        ax.scatter(s[column][latent], y[latent], c=list(colours[latent]), s=70, zorder=3, marker='D',
                   edgecolor=INK, linewidth=1.2)
        ax.set_title(title, fontsize=9, fontweight='normal')
        ax.grid(axis='y', visible=False)
        ax.margins(x=.08)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(s['setting'], fontsize=8.5)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=f'memory {m:g} ft')
               for m, c in C_MEM.items() if m in set(summary['memory_ft'])]
    handles.append(plt.Line2D([], [], marker='D', linestyle='', markerfacecolor=SURFACE, markeredgecolor=INK,
                              markersize=8, label='latent factor x3'))
    fig.legend(handles=handles, loc='outside upper right', ncols=4)
    fig.suptitle('Anisotropy scores, runs ranked by posterior Rv/Rh CRPS near the bit (lower is better)',
                 x=.01, ha='left', fontweight='bold', fontsize=11)
    fig.savefig(out / 'fig7_anisotropy.png')
    plt.close(fig)


def pick_runs(summary):
    """Base run plus the best run (posterior CRPS) of each memory length."""
    picks = []
    base = summary[summary['setting'] == 'mem 50 ft']
    if len(base):
        picks.append((base.index[0], INK2, (0, (4, 2))))
    for mem, group in summary.groupby('memory_ft'):
        name = group['post_crps_near'].idxmin()
        if name not in [p[0] for p in picks]:
            picks.append((name, C_MEM.get(mem, INK2), '-'))
    return picks


def fig_along_well(plt, out, summary, points):
    picks = pick_runs(summary)
    panels = [('forecast_chi2', 'Forecast misfit of the carried prior (chi2/n, log scale)', True),
              ('prior_rmse_near', 'Carried-prior Rh RMSE near bit (log10)', False),
              ('post_rmse_near', 'Posterior Rh RMSE near bit (log10)', False),
              ('post_z_rms_near', 'Posterior calibration z-rms near bit (1 = honest)', False)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 9.5), sharex=True, constrained_layout=True)
    gate = points[points['run'] == picks[0][0]]
    for ax, (column, title, log) in zip(axes, panels):
        for index in gate.loc[gate['gate_open'], 'index']:
            ax.axvspan(index - .5, index + .5, color='#f0efec', linewidth=0, zorder=0)
        for name, colour, dash in picks:
            run = points[points['run'] == name]
            ax.plot(run['index'], run[column], color=colour, linestyle=dash, label=summary.loc[name, 'setting'])
        if log:
            ax.set_yscale('log')
        if 'z_rms' in column:
            ax.axhline(1., color=INK2, linewidth=.8, linestyle=(0, (3, 2)))
        ax.set_title(title, fontsize=9.5)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside upper left', ncols=2, fontsize=8.5)
    axes[-1].set_xlabel('logging point  (shaded: gate open, point solved by the 2D stage)')
    fig.savefig(out / 'fig5_along_well.png')
    plt.close(fig)


def fig_profiles(plt, out, summary, profiles):
    picks = pick_runs(summary)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), sharex=True, constrained_layout=True)
    titles = [['Carried prior: error of the mean', 'Carried prior: ensemble spread'],
              ['Posterior: error of the mean', 'Posterior: ensemble spread']]
    for name, colour, dash in picks:
        stack = np.stack(list(profiles[name].values()))          # (points, 5, nx)
        xo = stack[0, 0]
        rms = np.sqrt((stack[:, [1, 3]] ** 2).mean(axis=0))
        std = stack[:, [2, 4]].mean(axis=0)
        for row in range(2):
            axes[row, 0].plot(xo, rms[row], color=colour, linestyle=dash, label=summary.loc[name, 'setting'])
            axes[row, 1].plot(xo, std[row], color=colour, linestyle=dash)
    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            ax.axvline(0, color=INK2, linewidth=.8)
            ax.set_title(titles[row][col], fontsize=9.5)
            ax.set_ylabel('log10 Rh')
        axes[1, row].set_xlabel('position relative to the bit (ft); positive = ahead')
    low, high = axes[0, 0].get_ylim()
    axes[0, 0].text(2, low + .02 * (high - low), 'bit', fontsize=8.5, color=INK2)
    axes[0, 0].legend(loc='upper left', fontsize=8)
    fig.suptitle(f'Error and spread across the window (rows within {ZONE_Z_FT:g} ft of the well, averaged over all points)',
                 x=.01, ha='left', fontweight='bold', fontsize=11)
    fig.savefig(out / 'fig6_profiles.png')
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path, nargs='?', default=Path(__file__).resolve().parent / 'runs')
    ap.add_argument('--glob', default='fault*', help="Run folders to compare (default: 'fault*')")
    ap.add_argument('--first', type=int, default=101,
                    help='First logging point scored (default 101: point 100 starts from the same prior everywhere)')
    ap.add_argument('--last', type=int, default=140)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'compare-output')
    ap.add_argument('--refresh', action='store_true', help='Recompute the per-run cache from the pickles')
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    settings, points, profiles = load_all(args.root, args.glob, args.out, args.first, args.last, args.refresh)
    summary = summarise(settings, points)
    effects = knob_effects(settings, points)
    points.to_csv(args.out / 'points.csv', index=False, float_format='%.6g')
    summary.to_csv(args.out / 'summary.csv', float_format='%.6g')
    effects.to_csv(args.out / 'knob_effects.csv', index=False, float_format='%.4g')

    pd.set_option('display.width', 250)
    show = ['setting', 'mean_memory_ft', 'post_crps_near', 'post_rmse_near', 'post_z_rms_near', 'post_cover90_near',
            'prior_crps_near', 'prior_crps_ahead', 'post_rv_crps_near', 'post_ratio_crps_near',
            'points_2d', 'map_hours', 'map_misfit']
    print(f'\n=== {len(summary)} runs, logging points {args.first}-{args.last}, ranked by posterior CRPS near the bit ===\n')
    print(summary[show].to_string(index=False, float_format='{:.3f}'.format))
    if len(effects):
        print('\n=== One-knob comparisons: % change going from -> to (negative = better; +/- paired s.e.) ===\n')
        table = effects[['knob', 'from', 'to', 'context']].copy()
        for column, name in (('post_crps_near', 'post CRPS'), ('post_rmse_near', 'post RMSE'),
                             ('post_z_rms_near', 'post z-rms'), ('prior_crps_ahead', 'prior CRPS ahead'),
                             ('post_rv_crps_near', 'post Rv CRPS'), ('post_ratio_crps_near', 'post Rv/Rh CRPS'),
                             ('prior_ratio_crps_near', 'prior Rv/Rh CRPS')):
            table[name] = [f'{m:+.1f} +/- {s:.1f}' for m, s in
                           zip(effects[f'd_{column}_pct'], effects[f'd_{column}_se_pct'])]
        print(table.to_string(index=False))

    if not args.no_figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        style(plt)
        fig_knobs(plt, args.out)
        fig_scoreboard(plt, args.out, summary)
        fig_mean_memory(plt, args.out, summary)
        fig_tradeoff(plt, args.out, summary)
        fig_along_well(plt, args.out, summary, points)
        fig_profiles(plt, args.out, summary, profiles)
        fig_anisotropy(plt, args.out, summary)
    print(f'\nWrote tables and figures to {args.out}')


if __name__ == '__main__':
    raise SystemExit(main())
