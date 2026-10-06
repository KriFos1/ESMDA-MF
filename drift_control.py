"""No-update drift control for the unified moving window.

Shifts an initial unified prior across the entire wellpath with conditional
edge entry but no assimilation, tracking ensemble statistics. Quantifies the
pure Monte-Carlo drift of the window machinery (spread collapse, mode-mix
drift, lateral-correlation decay) that the pilots can subtract from
data-driven diagnostics.
"""

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_DA_sequential as sequential
from esmda_core import LocalWindow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / 'Dashboard/benchmarks/Benchmark-3')
    parser.add_argument('--nx', type=int, default=33)
    parser.add_argument('--ensemble-size', type=int, default=250)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--stride', type=int, default=20)
    args = parser.parse_args(argv)
    sequential.dims = (args.nx, 5, 20)
    sequential.dX = sequential.dY = sequential.dZ = 5.0
    np.random.seed(args.seed)
    background, reservoir, mode_std = (float(np.log(1.5)), float(np.log(30.)), 0.2302585)
    zero_d = sequential.sample_prior_0D(args.ensemble_size, background, reservoir,
                                        mode_std, .15)[0][0]
    state = sequential.clamp_log_rh(sequential.sample_prior_2d(
        args.ensemble_size, zero_d, background, reservoir, mode_std, .15, .05,
        projection=False)[0])
    nx, _, nz = sequential.dims
    cells = nx * nz
    with h5py.File(args.benchmark / 'globalmodel.h5', 'r') as handle:
        wx = handle['wellpath/X'][:, 0] * sequential.meter_to_feet
        wy = handle['wellpath/Y'][:, 0] * sequential.meter_to_feet
        tvd = handle['wellpath/Z'][:, 0] * sequential.meter_to_feet
    window = LocalWindow(sequential.dims, (5., 5., 5.))

    def snapshot(tag):
        rh = state[:cells].reshape(nx, nz, -1)
        midpoint = .5 * (background + reservoir)
        reservoir_fraction = float((rh > midpoint).mean())
        bit = nx // 2
        correlations = {}
        for distance in (2, 8, 16):
            a = rh[bit].ravel(); b = rh[bit + distance].ravel()
            a -= a.mean(); b -= b.mean()
            correlations[distance] = float((a * b).mean()
                                            / np.sqrt((a * a).mean() * (b * b).mean()))
        return {'tag': tag, 'spread_ln': float(rh.std(axis=2).mean()),
                'reservoir_fraction': reservoir_fraction,
                'lateral_correlation': correlations}

    history = [snapshot('initial')]
    for index in range(len(wx)):
        _, movement = window.update((wx[index], wy[index], tvd[index]))
        state = sequential.clamp_log_rh(sequential.shift_window_state(
            state, movement, background, reservoir, mode_std,
            sequential.ratio_latent_std, .05))
        if index % args.stride == args.stride - 1:
            history.append(snapshot(f'index-{index + 1}'))
    history.append(snapshot('final'))
    report = {'configuration': {key: str(value) for key, value in vars(args).items()},
              'history': history}
    args.output.write_text(json.dumps(report, indent=2))
    print(f'Drift control: {args.output}')
    for entry in history[::max(1, len(history) // 10)]:
        print(json.dumps(entry))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
