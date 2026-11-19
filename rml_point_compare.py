#!/usr/bin/env python3
"""Compare the RML ensemble of one gated point (rml_point_check.py) with the chain's Laplace posterior.

Both use the same prior and data covariance. RML does not assume a single Gaussian, so:
  spread ratio RML / Laplace  ~ 1  -> the Laplace width is right for this prior and data;  >> 1 -> too narrow
  |mean shift| / Laplace std  ~ 0  -> the Laplace centre (the MAP) is representative
Both are also scored against the benchmark truth (z-rms, 5-95 % coverage), near the bit and over the window.

python rml_point_compare.py --chain runs/fault-mean-g25-s100 --rml runs/point-tests/rml/rml-idx115-fault-mean
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LN10 = np.log(10.)


def scores(fields, truth, mask):
    """fields: (cells, members) log10 Rh."""
    mean, std = fields.mean(axis=1), fields.std(axis=1, ddof=1)
    inside = (np.quantile(fields, .05, axis=1) <= truth) & (truth <= np.quantile(fields, .95, axis=1))
    z = (mean - truth) / std
    return {'rmse': float(np.sqrt(np.mean((mean - truth)[mask] ** 2))), 'spread': float(std[mask].mean()),
            'z_rms': float(np.sqrt(np.mean(z[mask] ** 2))), 'cover90': float(inside[mask].mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--rml', type=Path, required=True)
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location('compare_runs', ROOT / 'compare-runs.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)

    study = json.loads((args.rml / 'study.json').read_text())
    index, (nx, _, nz) = study['index'], study['dims']
    cells = nx * nz
    with open(args.chain / f'inversion_results_assim_{index}.pkl', 'rb') as stream:
        saved = pickle.load(stream)
    truth = compare.Truth(json.loads((args.chain / 'study.json').read_text())['benchmark']).window(saved)['rh'].ravel()
    laplace = np.asarray(saved['posterior_ensemble_params']).T[:cells] / LN10
    prior = np.asarray(saved['prior_params'])[:cells] / LN10

    models, rows = [], []
    for member in range(study['count']):
        folder = args.rml / f'member-{member:03d}'
        if not (folder / 'fit.npz').exists() or not (folder / 'result.json').exists():
            continue
        result = json.loads((folder / 'result.json').read_text())
        last = result['history'][-1]
        rows.append((member, result.get('reason'), result.get('accepted_iterations'), last['objective'],
                     last['likelihood_loss'], last['prior_loss']))
        with np.load(folder / 'fit.npz') as fit:
            models.append(fit['model'][:cells] / LN10)
    rml = np.array(models).T
    print(f'Index {index}: {rml.shape[1]} of {study["count"]} RML members finished; Laplace ensemble {laplace.shape[1]}')
    reasons = {}
    for row in rows:
        reasons[row[1]] = reasons.get(row[1], 0) + 1
    table = np.array([row[2:] for row in rows], dtype=float)
    print(f'  stop reasons {reasons}; iterations median {np.median(table[:, 0]):.0f}, max {table[:, 0].max():.0f}')
    print(f'  final objective median {np.median(table[:, 1]):.1f} (range {table[:, 1].min():.1f}-{table[:, 1].max():.1f}); '
          f'expected about {len(saved["observations"]) / 2:.0f} for a linear problem (half the number of data)')

    xo = (np.arange(nx) - nx // 2) * saved['spacing_ft'][0]
    zo = (np.arange(nz) - nz // 2) * saved['spacing_ft'][2]
    near = np.outer(np.abs(xo) <= compare.NEAR_X_FT, np.abs(zo) <= compare.ZONE_Z_FT).ravel()
    window = np.ones(cells, bool)
    print('\n                         RMSE   spread  z-rms  cover90   (log10 Rh)')
    for zone, mask in (('near bit', near), ('window', window)):
        for name, fields in (('carried prior', prior), ('Laplace', laplace), ('RML', rml)):
            s = scores(fields, truth, mask)
            print(f'  {zone:8s} {name:13s} {s["rmse"]:6.3f} {s["spread"]:7.3f} {s["z_rms"]:6.2f} {s["cover90"]:8.2f}')
    std_l, std_r = laplace.std(axis=1, ddof=1), rml.std(axis=1, ddof=1)
    shift = (rml.mean(axis=1) - laplace.mean(axis=1)) / std_l
    # Sampling error of a spread ratio from n members is about 1 / sqrt(2 (n - 1)) per cell.
    print(f'\n  spread ratio RML / Laplace: near bit median {np.median((std_r / std_l)[near]):.2f} '
          f'(10-90 %: {np.quantile((std_r / std_l)[near], .1):.2f}-{np.quantile((std_r / std_l)[near], .9):.2f}), '
          f'window median {np.median(std_r / std_l):.2f}; per-cell sampling error about '
          f'{1 / np.sqrt(2 * (rml.shape[1] - 1)):.2f}')
    print(f'  mean shift (RML mean - Laplace mean) / Laplace std: near bit rms {np.sqrt(np.mean(shift[near] ** 2)):.2f}, '
          f'window rms {np.sqrt(np.mean(shift ** 2)):.2f}; expected from sampling alone about '
          f'{np.median(std_r / std_l) / np.sqrt(rml.shape[1]):.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
