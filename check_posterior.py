"""Forward-validate saved 2D posterior members without changing the inversion."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import h5py
import numpy as np

from esmda_core import measurement_misfit
from esmda_remote2d import SimBackendConfig, make_remote2d
import run_DA_sequential as sequential


def main(argv=None):
    defaults = SimBackendConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    defaults.add_cli(parser)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'Dashboard/benchmarks/Benchmark-3')
    parser.add_argument('--solve-timeout', type=float, default=45.)
    args = parser.parse_args(argv)
    if not np.isfinite(args.solve_timeout) or args.solve_timeout <= 0:
        parser.error('Solve timeout must be finite and positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = args.checkpoint.read_bytes()
    result = pickle.loads(payload)
    if result['selected_fidelity'] != '2D':
        raise ValueError('Select a 2D checkpoint; saved 1D predictions are already forward evaluated')
    config = defaults.from_args(args)
    sequential.dims = tuple(result['dims'])
    dx, dy, dz = result['spacing_ft']
    adapter = make_remote2d(config, sequential.dims, dx, dy, dz, result['tools'])
    adapter.jobs_dir = str(output / 'jobs')
    with h5py.File(args.benchmark / 'globalmodel.h5', 'r') as handle:
        well = {key: value[:] for key, value in handle['wellpath'].items()}
    started = time.monotonic()
    report = {'checkpoint': str(args.checkpoint.resolve()),
              'checkpoint_sha256': hashlib.sha256(payload).hexdigest(),
              'logging_index': result['logging_index'], 'configuration': asdict(config),
              'prediction_kind': 'forward', 'members': []}
    try:
        predictions = adapter.run_pred_multi(
            [sequential._build_param_dict(param)[0] for param in result['posterior_params']],
            result['logging_index'], result['shift_ft'], well, args.solve_timeout)
        np.savez(output / 'forward.npz', predictions=predictions,
                 observation_mask=result['observation_mask'],
                 observations=result['observations'],
                 data_covariance=np.asarray(result['data_covariance']) / result['final_inflation_factor'])
        mask = result['observation_mask']
        covariance = np.asarray(result['data_covariance']) / result['final_inflation_factor']
        for member, prediction in enumerate(predictions):
            finite = bool(np.isfinite(prediction).all())
            report['members'].append({'member': member, 'passed': finite,
                                      'forward_misfit': measurement_misfit(
                                          prediction[mask], result['observations'], covariance) if finite else None,
                                      'stored_misfit': float(result['posterior_misfit'][member]),
                                      'stored_prediction_kind': result.get('posterior_prediction_kind', 'unknown')})
        report['passed'] = len(predictions) == len(result['posterior_params']) and all(
            member['passed'] for member in report['members'])
    except Exception as exc:
        report.update(passed=False, error=f'{type(exc).__name__}: {exc}')
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
