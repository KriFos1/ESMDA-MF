#!/usr/bin/env python3
"""Do the cells at layer boundaries of the truth dominate the posterior overconfidence?

Every window cell of every logging point is classed by its distance (cells, in any direction) to the
nearest jump of the truth, and interior cells by the layer they are in. Per class: share of cells,
error of the posterior mean, spread, z-rms, 5-95 % coverage and share of the summed squared z.

python boundary_check.py runs/my-run-4 runs/fault-mean-g25-s100
"""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent
JUMP = .1                       # log10 Rh difference between neighbours that counts as a boundary
RESISTIVE = 1.                  # log10 Rh above which an interior cell belongs to a resistive layer
CLASSES = ['at boundary', '1 cell away', '2 cells away', 'interior, resistive', 'interior, conductive']


def classify(truth):
    """Class index per cell of a (nx, nz) log10 Rh truth window."""
    edge = np.zeros(truth.shape, bool)
    for axis in (0, 1):
        jump = np.abs(np.diff(truth, axis=axis)) > JUMP
        lead = [slice(None)] * 2
        lag = [slice(None)] * 2
        lead[axis], lag[axis] = slice(1, None), slice(None, -1)
        edge[tuple(lead)] |= jump
        edge[tuple(lag)] |= jump
    square = np.ones((3, 3), bool)
    one = binary_dilation(edge, square)
    two = binary_dilation(one, square)
    out = np.where(truth > RESISTIVE, 3, 4)
    out[two], out[one], out[edge] = 2, 1, 0
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location('compare_runs', ROOT / 'compare-runs.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)

    for folder in args.runs:
        study = json.loads((folder / 'study.json').read_text())
        truth = compare.Truth(study['benchmark'])
        two_d = {r['logging_index'] for r in json.loads((folder / 'history.json').read_text())
                 if r.get('stage') == 'laplace_2d'}
        store = {}                                    # (stage, zone) -> lists of class, err, std, inside
        for path in sorted(folder.glob('inversion_results_assim_*.pkl')):
            index = int(path.stem.split('_')[-1])
            with open(path, 'rb') as stream:
                saved = pickle.load(stream)
            nx, _, nz = saved['dims']
            dx, _, dz = saved['spacing_ft']
            target = truth.window(saved)['rh']
            field = np.asarray(saved['posterior_ensemble_params']).T[:nx * nz].reshape(nx, nz, -1) / compare.LN10
            mean, std = field.mean(axis=2), field.std(axis=2, ddof=1)
            inside = (np.quantile(field, .05, axis=2) <= target) & (target <= np.quantile(field, .95, axis=2))
            kind = classify(target)
            xo, zo = (np.arange(nx) - nx // 2) * dx, (np.arange(nz) - nz // 2) * dz
            zones = {'near bit': np.outer(np.abs(xo) <= compare.NEAR_X_FT, np.abs(zo) <= compare.ZONE_Z_FT),
                     'window': np.ones((nx, nz), bool)}
            for stage in ('all', '2D' if index in two_d else '1D'):
                for zone, mask in zones.items():
                    store.setdefault((stage, zone), []).append(
                        np.stack([kind[mask], (mean - target)[mask], std[mask], inside[mask]]))
        print(f'\n=== {folder.name} ===')
        for (stage, zone), parts in sorted(store.items()):
            kind, err, std, inside = np.concatenate(parts, axis=1)
            z2 = (err / std) ** 2
            print(f'\n{zone}, {stage} points ({len(parts)}): all cells z-rms {np.sqrt(z2.mean()):.2f}, '
                  f'coverage {inside.mean():.2f}')
            print('  class                  cells   error  spread  z-rms  cover90  share of z^2')
            for number, name in enumerate(CLASSES):
                m = kind == number
                if not m.any():
                    continue
                print(f'  {name:21s} {m.mean():6.1%} {np.sqrt(np.mean(err[m] ** 2)):7.3f} {std[m].mean():7.3f} '
                      f'{np.sqrt(z2[m].mean()):6.2f} {inside[m].mean():8.2f} {z2[m].sum() / z2.sum():10.0%}')
            away = kind >= 3
            print(f'  -> 3 or more cells from any boundary: z-rms {np.sqrt(z2[away].mean()):.2f}, '
                  f'coverage {inside[away].mean():.2f} ({away.mean():.0%} of cells)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
