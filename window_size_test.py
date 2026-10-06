"""Window-size and cell-coarsening test: forward boundary error of truth windows.

Each candidate (nx, dx, nz, dz) cell-averages the truth model into the window
centered at chosen logging points and evaluates it with the 2D forward
simulator. Responses are compared against a large fine reference window in
measurement-noise units (per tool, angle channels wrapped), isolating:

- boundary truncation error at 5 ft cells (nested fine windows);
- the extra error of 10 ft lateral cells (same-or-better reach, coarse cells);
- vertical coarsening (10 ft rows) as a probe.

Solver seconds per field are recorded for cost scaling. The smallest
configuration whose boundary-induced response error stays below the noise
level on every tool is the recommendation candidate.
"""

import argparse
import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm

import run_DA_sequential as sequential
from esmda_core import LocalWindow
from esmda_remote2d import SimBackendConfig, make_remote2d

CONFIGS = [  # name, nx, dx_ft, nz, dz_ft
    ('reference-41x5', 41, 5., 20, 5.),
    ('check-33x5', 33, 5., 20, 5.),
    ('w15x5', 15, 5., 20, 5.),
    ('w21x5', 21, 5., 20, 5.),
    ('w27x5', 27, 5., 20, 5.),
    ('w15x10', 15, 10., 20, 5.),
    ('w21x10', 21, 10., 20, 5.),
    ('w15x5z10', 15, 5., 20, 10.),
    ('g15x5-coarse-grid', 15, 5., 20, 5.),
]
REFERENCE = 'reference-41x5'


def matched_grid(nx, nz):
    """Uniform nodes covering the window cells; no material averaging.

    Nx_uni must be divisible by 4 and Nz_uni even (fwd2D input contract);
    optimal nodes concentrate near the tool and are kept at wrapper defaults.
    """
    return (int(4 * np.ceil(nx / 4)), 4, 2 * int(np.ceil(nz / 2)), 30, 20, 30, 1, 1, 1)


def overlap_weights(centers_ft, edges_ft, cell_ft):
    """Normalized overlap of uniform truth cells with each window cell."""
    centers_ft = np.asarray(centers_ft, dtype=float)
    low, high = centers_ft - cell_ft / 2, centers_ft + cell_ft / 2
    weights = np.empty((len(edges_ft) - 1, len(centers_ft)))
    for index in range(weights.shape[0]):
        span = np.maximum(0., np.minimum(edges_ft[index + 1], high)
                          - np.maximum(edges_ft[index], low))
        weights[index] = span / (high - low)
    totals = weights.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError('Window cell does not overlap the truth grid')
    return weights / totals


def truth_window(handle, shift_ft, nx, dx, nz, dz):
    """Overlap-weighted cell averages of truth Rh/Rv as (ln Rh, latent) fields."""
    mid = handle['formation/Rh'].shape[1] // 2
    x_t = handle['formation/x'][:, 0] / 0.3048
    z_t = handle['formation/z'][:, 0] / 0.3048
    x_edges = shift_ft['x'] + np.arange(nx + 1) * dx
    z_edges = shift_ft['z'] + np.arange(nz + 1) * dz
    wx = overlap_weights(x_t, x_edges, float(np.median(np.diff(x_t))))
    wz = overlap_weights(z_t, z_edges, float(np.median(np.diff(z_t))))
    fields = {}
    for name in ('Rh', 'Rv'):
        truth = handle[f'formation/{name}'][:, mid, :]
        averaged = (wz @ (wx @ truth.T).T).T
        fields[name] = np.maximum(averaged, 1e-3)
    log_rh = np.log(fields['Rh'])
    log_ratio = np.clip(np.log(fields['Rv']) - log_rh,
                        np.log(sequential.min_ratio), np.log(sequential.max_ratio))
    probability = np.clip((log_ratio - np.log(sequential.min_ratio))
                          / np.log(sequential.max_ratio / sequential.min_ratio), 1e-6, 1 - 1e-6)
    latent = norm.ppf(probability)
    return np.r_[log_rh.ravel(), latent.ravel()]


def read_sigma(variance_path, index):
    variances = pd.read_pickle(variance_path)
    return np.concatenate([np.asarray(entry[1], dtype=float)
                           for entry in variances.iloc[index]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / 'Dashboard/benchmarks/Benchmark-3')
    parser.add_argument('--variance', type=Path,
                        default=Path(__file__).resolve().parent
                        / 'runs/benchmark3-data-20261002/var.pkl')
    parser.add_argument('--indices', type=int, nargs='+', default=[27, 80])
    defaults = SimBackendConfig.from_env()
    defaults.add_cli(parser)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_cwd = Path.cwd()
    benchmark = args.benchmark.resolve()
    tools = [str(tool) for tool in pd.read_pickle(args.variance).columns]
    tool_pairs = [ast.literal_eval(tool) for tool in tools]
    with h5py.File(benchmark / 'globalmodel.h5', 'r') as handle:
        wellpath = {key: value[:] for key, value in handle['wellpath'].items()}
        positions = {index: (wellpath['X'][index, 0] * sequential.meter_to_feet,
                             wellpath['Y'][index, 0] * sequential.meter_to_feet,
                             wellpath['Z'][index, 0] * sequential.meter_to_feet)
                     for index in args.indices}
    config = defaults.from_args(args)
    report = {'configuration': {name: dict(zip(('nx', 'dx_ft', 'nz', 'dz_ft'), values))
                                for name, *values in CONFIGS},
              'reference': REFERENCE, 'indices': args.indices,
              'simulator': json.loads(json.dumps(vars(args), default=str)),
              'results': {}, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    os.chdir(output)
    try:
        responses = {}
        for name, nx, dx, nz, dz in CONFIGS:
            dims = (nx, 5, nz)
            sequential.dims = dims
            sequential.dX, sequential.dY, sequential.dZ = dx, dx, dz
            grid = (20, 4, 40, 30, 20, 30, 2, 1, 2) if 'coarse-grid' in name \
                else matched_grid(nx, nz)
            case_config = SimBackendConfig(**{**asdict(config), 'solver_grid': grid})
            for index in args.indices:
                window = LocalWindow(dims, (dx, dx, dz))
                shift, _ = window.update(np.asarray(positions[index]))
                adapter = make_remote2d(case_config, dims, dx, dx, dz, tool_pairs)
                jobs = output / 'jobs' / f'{name}-{index}'
                adapter.jobs_dir = str(jobs)
                parameters = truth_window_of(args.benchmark, shift, nx, dx, nz, dz)
                param_dict, _, _ = sequential._build_param_dict(parameters)
                started = time.monotonic()
                prediction = adapter.run_pred(param_dict, index, shift, wellpath)
                responses[(name, index)] = np.asarray(prediction, dtype=float)
                report['results'][f'{name}-{index}'] = {
                    'client_seconds': time.monotonic() - started, 'solver_grid': list(grid)}
                print(f'{name} index {index}: evaluated', flush=True)
        np.savez(output / 'responses.npz',
                 **{f'{name}-{index}': value for (name, index), value in responses.items()})
        for index in args.indices:
            sigma = read_sigma(args.variance, index)
            usable = sigma >= 1e-3
            reference = responses[(REFERENCE, index)]
            for name, *_ in CONFIGS:
                if name == REFERENCE:
                    continue
                delta = responses[(name, index)] - reference
                delta[::2] = (delta[::2] + 180) % 360 - 180
                normalized = np.abs(delta) / sigma
                entry = report['results'][f'{name}-{index}']
                per_tool = {tool: {'max_sigma': float(normalized[8*t:8*t+8][usable[8*t:8*t+8]].max()),
                                   'mean_sigma': float(normalized[8*t:8*t+8][usable[8*t:8*t+8]].mean())}
                            for t, tool in enumerate(tools)}
                entry.update(max_sigma=float(normalized[usable].max()),
                             mean_sigma=float(normalized[usable].mean()),
                             masked_channels=int((~usable).sum()), per_tool=per_tool)
    finally:
        os.chdir(previous_cwd)
        (output / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Assessment: {output / "assessment.json"}')
    for index in args.indices:
        print(f'== index {index} (max |delta| / sigma per tool) ==')
        for name, *_ in CONFIGS:
            if name == REFERENCE:
                continue
            entry = report['results'].get(f'{name}-{index}', {})
            if 'max_sigma' in entry:
                worst = max(entry['per_tool'].items(), key=lambda item: item[1]['max_sigma'])
                print(f'{name:14s} overall {entry["max_sigma"]:7.2f}  worst {worst[0]} '
                      f'{worst[1]["max_sigma"]:.2f}')
    return 0


def truth_window_of(benchmark, shift, nx, dx, nz, dz):
    with h5py.File(Path(benchmark) / 'globalmodel.h5', 'r') as handle:
        return truth_window(handle, shift, nx, dx, nz, dz)


if __name__ == '__main__':
    raise SystemExit(main())
