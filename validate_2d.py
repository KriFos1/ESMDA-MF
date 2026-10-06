"""Validate the full local-window UDAR Jacobian against directional differences.

Optionally checks 1D/2D fidelity consistency: a homogeneous-in-x 2D tile built
from a layered model must reproduce the UTA1D prediction of the same model.
"""

from dataclasses import asdict
import argparse
import ast
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import run_DA_sequential as sequential
from esmda_core import LocalWindow
from esmda_remote2d import SimBackendConfig, make_remote2d

DEFAULT_DATA = Path(__file__).resolve().parent / 'runs/benchmark3-data-20261002/data.pkl'


def parse_tools(values):
    if not values:
        return [str(column) for column in pd.read_pickle(DEFAULT_DATA).columns]
    tools = []
    for value in values:
        freq, sub = value.split(',')
        tools.append(f"('{freq.strip()}','{sub.strip()}')")
    return tools


def synthetic_base(nx, nz):
    x, z = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')
    log_rh = np.log(10.) + .3 * np.tanh((z - nz / 2) / 3) + .07 * np.sin(x * .4)
    latent = .2 * np.cos(z * .3) + .1 * np.sin(x * .5)
    return log_rh, latent


def anisotropy_base(nx, nz):
    x, z = np.meshgrid(np.arange(nx), np.arange(nz), indexing='ij')
    log_rh = np.log(10.) + .2 * np.tanh((z - nz / 2) / 4)
    latent = 1.5 * np.tanh((x - nx / 2) / 2) + .1 * np.cos(z * .3)
    return log_rh, latent


def posterior_column(posterior_path, index, member=0):
    result = pd.read_pickle(Path(posterior_path) / f'inversion_results_assim_{index}.pkl')
    column = np.asarray(result['post_1d_params'])[:, member]
    nz = len(column) // 2
    return column[:nz], column[nz:]


def parse_args(argv=None):
    defaults = SimBackendConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    defaults.add_cli(parser)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--steps', type=float, nargs='+', default=[.003, .001],
                        help='Central-difference steps (avoid cancellation in solver LAS output)')
    parser.add_argument('--tools', nargs='+', default=None,
                        help='Tool pairs as FREQ,SUB (default: benchmark-3 data columns)')
    parser.add_argument('--nx', type=int, default=15)
    parser.add_argument('--ny', type=int, default=5)
    parser.add_argument('--nz', type=int, default=20)
    parser.add_argument('--dx-ft', type=float, default=5.)
    parser.add_argument('--dz-ft', type=float, default=5.)
    parser.add_argument('--base', choices=['synthetic', 'anisotropy', 'posterior', 'homogeneous',
                                           'step'],
                        default='synthetic')
    parser.add_argument('--rh-value', type=float, default=1.0,
                        help='Constant log10 Rh for --base homogeneous/step')
    parser.add_argument('--latent-value', type=float, default=0.0,
                        help='Constant anisotropy latent for --base homogeneous/step')
    parser.add_argument('--step-at', type=int, default=10,
                        help='Boundary cell for --base step')
    parser.add_argument('--step-size', type=float, default=1.0,
                        help='log10 Rh contrast across the boundary for --base step')
    parser.add_argument('--latent-step-size', type=float, default=0.0,
                        help='Anisotropy latent contrast across the boundary for --base step')
    parser.add_argument('--posterior-run', type=Path, default=None,
                        help='Sequential run directory for --base posterior')
    parser.add_argument('--posterior-index', type=int, default=27)
    parser.add_argument('--trajectory-index', type=int, default=27,
                        help='Logging point whose geometry the window is centred on')
    parser.add_argument('--consistency', action='store_true',
                        help='Compare an x-invariant tile against the 1D simulator '
                             '(only meaningful for laterally constant bases: '
                             'homogeneous, step, posterior)')
    parser.add_argument('--skip-fd', action='store_true',
                        help='Skip the directional finite-difference checks')
    args = parser.parse_args(argv)
    if len(args.steps) < 2 or len(set(args.steps)) != len(args.steps) \
            or not np.isfinite(args.steps).all() or min(args.steps) <= 0:
        parser.error('Use at least two distinct positive finite-difference steps')
    if args.base == 'posterior' and args.posterior_run is None:
        parser.error('--base posterior requires --posterior-run')
    if min(args.nx, args.ny, args.nz, args.dx_ft, args.dz_ft) <= 0:
        parser.error('Window dimensions and cell sizes must be positive')
    return args, defaults.from_args(args)


def main(argv=None):
    args, config = parse_args(argv)
    tools = parse_tools(args.tools)
    output = args.output_dir.resolve()
    posterior_run = (args.posterior_run.resolve() if args.posterior_run else None)
    output.mkdir(parents=True, exist_ok=False)
    previous_cwd = Path.cwd()
    report = {'configuration': None, 'runtime': None, 'checks': [], 'step_consistency': [],
              'consistency': None, 'dims': (args.nx, args.ny, args.nz), 'tools': tools,
              'base': args.base, 'trajectory_index': args.trajectory_index,
              'parameterization': '[ln(Rh), anisotropy latent]', 'seed': args.seed,
              'steps': args.steps, 'relative_l2_limit': .02, 'passed': False}
    os.chdir(output)
    try:
        sequential.dims = (args.nx, args.ny, args.nz)
        sequential.dX = sequential.dY = args.dx_ft
        sequential.dZ = args.dz_ft
        sequential.tools = tools
        nx, ny, nz = sequential.dims
        n_cells = nx * nz
        if args.base == 'synthetic':
            log_rh, latent = synthetic_base(nx, nz)
        elif args.base == 'anisotropy':
            log_rh, latent = anisotropy_base(nx, nz)
        elif args.base == 'homogeneous':
            log_rh = np.full((nx, nz), args.rh_value)
            latent = np.full((nx, nz), args.latent_value)
        elif args.base == 'step':
            log_rh = np.full((nx, nz), args.rh_value)
            log_rh[:, args.step_at:] += args.step_size
            latent = np.full((nx, nz), args.latent_value)
            latent[:, args.step_at:] += args.latent_step_size
        else:
            column_rh, column_latent = posterior_column(posterior_run, args.posterior_index)
            if len(column_rh) != nz:
                raise ValueError(f'Posterior column has {len(column_rh)} layers, window has {nz}')
            log_rh = np.tile(column_rh[None, :], (nx, 1))
            latent = np.tile(column_latent[None, :], (nx, 1))
        parameters = np.r_[log_rh.ravel(), latent.ravel()]
        index = args.trajectory_index
        benchmark = Path(sequential.parse_args(['--output-dir', 'unused'])[0].benchmark).resolve()
        with h5py.File(benchmark / 'globalmodel.h5', 'r') as handle:
            wellpath = {key: value[:] for key, value in handle['wellpath'].items()}
        position = (wellpath['X'][index, 0] * sequential.meter_to_feet,
                    wellpath['Y'][index, 0] * sequential.meter_to_feet,
                    wellpath['Z'][index, 0] * sequential.meter_to_feet)
        window = LocalWindow(sequential.dims, (sequential.dX, sequential.dY, sequential.dZ))
        sequential.current_shift, _ = window.update(np.asarray(position))
        sequential.wellpath = wellpath
        sequential.remote_2d = make_remote2d(config, sequential.dims, sequential.dX,
                                             sequential.dY, sequential.dZ,
                                             [ast.literal_eval(tool) for tool in tools])
        sequential.remote_2d.jobs_dir = str(output / 'jobs')
        runtime = sequential.remote_2d.runtime_identity()
        report.update(configuration=asdict(config), runtime=runtime,
                      dims=sequential.dims)
        prediction, jacobian = sequential.simulate_and_grad(parameters, index)
        forward = sequential.simulate_pred_only(parameters, index)
        np.testing.assert_allclose(prediction, forward, rtol=1e-7, atol=1e-8)
        n_channels = 8 * len(tools)
        if prediction.shape != (n_channels,) or jacobian.shape != (n_channels, 2 * n_cells) \
                or not np.isfinite(jacobian).all():
            raise RuntimeError('Invalid full-window prediction/Jacobian')
        np.savez(output / 'base.npz', parameters=parameters, prediction=prediction, jacobian=jacobian)
        rng = np.random.default_rng(args.seed)
        if not args.skip_fd:
            for block, name in enumerate(('lnRh', 'anisotropy_latent')):
                for kind in ('cell', 'direction'):
                    direction = np.zeros(2 * n_cells)
                    if kind == 'cell':
                        direction[block * n_cells + (nx // 2) * nz + nz // 2] = 1
                    else:
                        direction[block*n_cells:(block+1)*n_cells] = rng.normal(size=n_cells)
                        direction /= np.linalg.norm(direction)
                    analytic = jacobian @ direction
                    differences = []
                    for step in args.steps:
                        plus = sequential.simulate_pred_only(parameters + step * direction, index)
                        minus = sequential.simulate_pred_only(parameters - step * direction, index)
                        difference = plus - minus
                        difference[::2] = (difference[::2] + 180) % 360 - 180
                        finite_difference = difference / (2 * step)
                        relative = np.linalg.norm(analytic - finite_difference) \
                            / max(np.linalg.norm(finite_difference), 1e-15)
                        record = {'parameter': name, 'kind': kind, 'step': step,
                                  'relative_l2': float(relative), 'passed': bool(relative <= .02),
                                  'analytic': analytic.tolist(),
                                  'finite_difference': finite_difference.tolist()}
                        report['checks'].append(record)
                        differences.append(finite_difference)
                        print(f'{name} {kind} h={step}: relative L2={relative:.6g}', flush=True)
                    for i in range(len(differences) - 1):
                        relative = np.linalg.norm(differences[i] - differences[i+1]) \
                            / max(np.linalg.norm(differences[i+1]), 1e-15)
                        report['step_consistency'].append(
                            {'parameter': name, 'kind': kind, 'steps': args.steps[i:i+2],
                             'relative_l2': float(relative), 'passed': bool(relative <= .02)})
        report['passed'] = (not report['checks']) or all(
            check['passed'] for check in report['checks'] + report['step_consistency'])
        if args.consistency:
            report['consistency'] = check_consistency(args, tools, index, benchmark, parameters)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        os.chdir(previous_cwd)
    return 0 if report['passed'] else 1


def check_consistency(args, tools, index, benchmark, parameters):
    sequential.setup_simulators(benchmark, tools)
    sequential._set_logging_point_state(index)
    nx, nz = args.nx, args.nz
    prediction_2d = sequential.simulate_pred_only(parameters, index)
    center = nx // 2
    column = np.r_[parameters[center * nz:(center + 1) * nz],
                   parameters[nx * nz + center * nz:nx * nz + (center + 1) * nz]]
    prediction_1d = sequential.simulate_pred_only(column, index, '1D')
    per_tool = []
    for t, tool in enumerate(tools):
        block = slice(8 * t, 8 * (t + 1))
        rel = float(np.linalg.norm(prediction_2d[block] - prediction_1d[block])
                    / max(np.linalg.norm(prediction_1d[block]), 1e-15))
        per_tool.append({'tool': str(tool), 'relative_l2': rel})
        print(f'consistency {tool}: 1D vs 2D relative L2={rel:.6g}', flush=True)
    return {'trajectory_index': index, 'per_tool': per_tool,
            'max_relative_l2': max(entry['relative_l2'] for entry in per_tool)}


if __name__ == '__main__':
    raise SystemExit(main())
