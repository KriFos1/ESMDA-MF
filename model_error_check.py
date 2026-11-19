#!/usr/bin/env python3
"""How large is the 1D model error, and how much of it is the 5 ft grid? (local, no inversion)

For every logging point the measured (benchmark, 2D) data are compared with the 1D response of the true
column under the bit, once with the truth averaged onto the inversion grid (5 ft cells) and once on a
fine vertical grid (0.5 ft, the resolution of the benchmark model). In units of the data error model:

  data - coarse    what the 1D stage faces: grid effect and real 2D effects together
  data - fine      real 2D effects only (the 1D model is exact for flat layers of any thickness)
  coarse - fine    the grid effect alone

python model_error_check.py --chain runs/my-run-4 --out compare-output/model-error
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_DA_sequential as sequential
from esmda_core import LocalWindow, align_phases, noise_floor_mask, observation_vectors, phase_channels
from esmda_remote2d import UDAR_COMPONENTS
from sequential_matern_1d import RV_CAP_OHMM, directional_floor

ROOT = Path(__file__).resolve().parent
LN10 = np.log(10.)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, default=ROOT / 'runs/benchmark3-reference-bedding')
    parser.add_argument('--chain', type=Path, default=ROOT / 'runs/my-run-4',
                        help='Chain whose history tells which points were accepted as 1D')
    parser.add_argument('--out', type=Path, default=ROOT / 'compare-output/model-error')
    parser.add_argument('--fine-dz', type=float, default=.5)
    parser.add_argument('--directional-sigma-floor', type=float, nargs=2, default=(.5, .25))
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location('compare_runs', ROOT / 'compare-runs.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)

    manifest = json.loads((args.reference.resolve() / 'run.json').read_text())
    nx, ny, nz = manifest['dims']
    dx, dy, dz = manifest['spacing_ft']
    refine = int(round(dz / args.fine_dz))
    bit = nx // 2
    data = pd.read_pickle(ROOT / manifest['data'])
    variances = pd.read_pickle(ROOT / manifest['variance'])
    tool_pairs = list(data.columns)
    sequential.tools = [str(tool) for tool in tool_pairs]
    sequential.max_ratio = float(manifest['prior_ratio_max'])
    sequential.log_rv_cap = float(np.log(RV_CAP_OHMM))
    sequential.anisotropy = 'log_clipped'
    truth_model = compare.Truth(manifest['benchmark'])
    history = json.loads((args.chain / 'history.json').read_text())
    gated = {record['logging_index'] for record in history if record.get('stage') is not None}
    last = max(record['logging_index'] for record in history)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def response(index, shift, cells, spacing):
        """1D response of the truth column on a vertical grid of ``cells`` cells of ``spacing`` ft."""
        sequential.dims = (nx, ny, cells)
        sequential.dX, sequential.dY, sequential.dZ = dx, dy, spacing
        sequential.current_shift = shift
        target = truth_model.window({'dims': (nx, ny, cells), 'spacing_ft': (dx, dy, spacing),
                                     'shift_ft': dict(shift)})
        column = np.concatenate([target['rh'][bit], target['ratio'][bit]]) * LN10
        return sequential.simulate_pred_only(column, index, '1D')

    rows, channels = [], []
    previous_cwd = Path.cwd()
    try:
        os.chdir(out)
        sequential.setup_simulators(Path(manifest['benchmark']), sequential.tools)
        window = LocalWindow((nx, ny, nz), (dx, dy, dz))
        for index in range(last + 1):
            shift, _ = window.update((sequential.WX[index], sequential.WY[index], sequential.TVD[index]))
            observations, variance = observation_vectors(data, variances, index, tool_pairs)
            variance = directional_floor(variance, args.directional_sigma_floor)
            mask = ~noise_floor_mask(observations, variance)
            phase = phase_channels(np.ones(len(observations), bool), UDAR_COMPONENTS)
            coarse = align_phases(response(index, shift, nz, dz), observations, phase)
            fine = align_phases(response(index, shift, nz * refine, dz / refine), observations, phase)
            sigma = np.sqrt(variance)
            parts = {'data_coarse': (observations - coarse) / sigma, 'data_fine': (observations - fine) / sigma,
                     'coarse_fine': (coarse - fine) / sigma}
            rows.append({'index': index, 'stage': '2D' if index in gated else '1D', 'channels': int(mask.sum()),
                         **{name: float(np.mean(value[mask] ** 2)) for name, value in parts.items()}})
            for position in np.flatnonzero(mask):
                channels.append({'index': index, 'stage': rows[-1]['stage'],
                                 'tool': '/'.join(tool_pairs[position // len(UDAR_COMPONENTS)]),
                                 'component': UDAR_COMPONENTS[position % len(UDAR_COMPONENTS)],
                                 'sigma': sigma[position], 'observation': observations[position],
                                 **{name: float(value[position]) for name, value in parts.items()}})
    finally:
        os.chdir(previous_cwd)
    table, detail = pd.DataFrame(rows), pd.DataFrame(channels)
    table.to_csv(out / 'points.csv', index=False)
    detail.to_csv(out / 'channels.csv', index=False)

    names = ['data_coarse', 'data_fine', 'coarse_fine']
    print('Mismatch per datum (mean squared difference in units of the data error variance); median over points')
    print('  points        stage  count   data-coarse   data-fine  coarse-fine')
    groups = [('all', table)] + [(f'{lo}-{hi}', table[(table['index'] >= lo) & (table['index'] <= hi)])
                                 for lo, hi in ((0, 23), (24, 68), (69, 102), (103, 133), (134, last))]
    for label, group in groups:
        for stage, part in [('any', group)] + list(group.groupby('stage')):
            if len(part):
                print(f'  {label:12s} {stage:6s} {len(part):5d} ' + ' '.join(f'{part[n].median():12.1f}' for n in names))
    print('\nRoot mean squared mismatch in units of the data error, by component (all points)')
    by = detail.groupby('component')[names].agg(lambda v: np.sqrt(np.mean(v ** 2))).reindex(UDAR_COMPONENTS)
    print(by.round(1).to_string())
    print('\n... and by tool')
    print(detail.groupby('tool')[names].agg(lambda v: np.sqrt(np.mean(v ** 2))).round(1).to_string())
    print(f'\nTables: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
