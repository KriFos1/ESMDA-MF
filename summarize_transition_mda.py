"""Summarize the matched threshold × 2D-step study without censoring failures."""

import argparse
import json
from pathlib import Path

import numpy as np

from transition_mda_study import INDICES, THRESHOLDS


def summarize(output):
    points = [json.loads((output / f'point-{i}.json').read_text()) for i in INDICES]
    table = []
    for threshold in THRESHOLDS:
        for steps in range(6):
            selected = [(point, point['branches'][str(steps)]
                         if point['gate_misfit'] > threshold
                         else point['branches']['0']) for point in points]
            valid = [(point, row) for point, row in selected if row['status'] == 'ok']
            gated = [row for point, row in selected if point['gate_misfit'] > threshold
                     and steps > 0]
            misfits = [row['data_misfit'] for _, row in valid]
            table.append({'threshold': threshold, '2d_steps': steps,
                          'gated_points': sum(point['gate_misfit'] > threshold for point in points),
                          'valid_points': len(valid),
                          'reliable_gated_points': sum(row['status'] == 'ok'
                                                       and row['survivors'] >= 20 for row in gated),
                          'min_gated_survivors': (min(row['survivors'] for row in gated)
                                                  if gated else None),
                          'failed_points': [point['index'] for point, row in selected
                                            if row['status'] != 'ok'],
                          'mean_rmse_log10_rh': (float(np.mean([row['rmse_log10_rh']
                                                              for _, row in valid]))
                                                 if len(valid) == len(points) else None),
                          'mean_reduced_misfit': (float(np.mean(misfits))
                                                  if len(valid) == len(points) else None),
                          'points_misfit_le_75': sum(value <= 75. for value in misfits),
                          '2d_calls': sum(row['2d_calls'] for _, row in selected)})
    best = []
    for point in points:
        acceptable = [step for step in range(6)
                      if point['branches'][str(step)]['status'] == 'ok'
                      and point['branches'][str(step)]['data_misfit'] <= 75.]
        best.append({'index': point['index'], 'gate_misfit': point['gate_misfit'],
                     'first_step_reaching_75': min(acceptable) if acceptable else None,
                     'one_d_rmse': point['branches']['0']['rmse_log10_rh'],
                     'one_d_misfit': point['branches']['0']['data_misfit'],
                     'two_d_failures': {str(k): point['branches'][str(k)]['failed_evaluations']
                                        for k in range(1, 6)}})
    total_calls = sum(point['branches'][str(k)]['2d_calls']
                      for point in points for k in range(1, 6))
    failed_evaluations = sum(len(point['branches'][str(k)]['failed_evaluations'])
                             for point in points for k in range(1, 6))
    return {'table': table, 'points': best,
            'total_2d_calls': total_calls, 'failed_2d_evaluations': failed_evaluations,
            'misfit_goal': 'Original uninflated reduced chi-square <= 75; '
                           'no failed point may contribute to a scored configuration; '
                           '>=20/25 survivors at every gated point for reliable comparison',
            'interpretation': 'Independent pointwise replay, not a chained inversion; '
                              'thresholds evaluated on the shared step-5 1D ensemble.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    result = summarize(output)
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    lines = ['# Ten-step transition ES-MDA: matched threshold and 2D-step study', '',
             result['interpretation'], '', result['misfit_goal'], '',
             f"HPC evaluations: {result['total_2d_calls']} 2D forwards, "
             f"{result['failed_2d_evaluations']} non-finite solver evaluations. "
             'The same branches are reused for all six offline thresholds.', '',
             '| Gate | 2D steps | Gated points | Valid / 16 | Reliable gated | '
             'Min 2D survivors | Mean Rh RMSE (log10) | '
             'Mean reduced misfit | Points <=75 | HPC 2D calls |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['table']:
        fmt = lambda value: f'{value:.4g}' if value is not None else 'invalid'
        lines.append(f"| {row['threshold']:g} | {row['2d_steps']} | {row['gated_points']} | "
                     f"{row['valid_points']}/16 | {row['reliable_gated_points']} | "
                     f"{row['min_gated_survivors'] if row['min_gated_survivors'] is not None else '-'} | "
                     f"{fmt(row['mean_rmse_log10_rh'])} | "
                     f"{fmt(row['mean_reduced_misfit'])} | {row['points_misfit_le_75']} | "
                     f"{row['2d_calls']} |")
    lines.extend(['', '## Pointwise minimum steps reaching reduced misfit <= 75', '',
                  '| Index | Step-5 1D gate score | 1D final misfit | '
                  '1D Rh RMSE | First 2D steps (0–5) |',
                  '|---:|---:|---:|---:|---:|'])
    for row in result['points']:
        first = row['first_step_reaching_75']
        lines.append(f"| {row['index']} | {row['gate_misfit']:.4g} | "
                     f"{row['one_d_misfit']:.4g} | {row['one_d_rmse']:.4g} | "
                     f"{first if first is not None else 'none'} |")
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(output / 'summary.md')


if __name__ == '__main__':
    main()
