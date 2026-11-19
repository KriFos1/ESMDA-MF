#!/usr/bin/env python3
"""Per-channel model-error factors from the truth residuals of the benchmark (model_error_check.py and
truth_forward_2d.py), for sequential_matern_1d.py --model-error.

factor[tool][component] = standard deviation of the model error in units of the measurement standard
deviation. It is the root mean squared mismatch between the data and the response of the TRUE model,
taken over the calm layered points (no fault within the window) and fitted as a product of a tool
factor and a component factor, which keeps the table stable when few points are available:

  1D stage: 1D response of the true bit column, points the chain accepted as 1D inside the thick layer
  2D stage: 2D response of the truth on the 5 ft window grid, at the layered points with a 2D check

The term needs the truth, so it is a benchmark calibration of how large the model error is and which
channels carry it, not something available in the field.

python build_model_error.py            # writes references/model-error-benchmark3{,-1d,-2d}.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from esmda_remote2d import UDAR_COMPONENTS

ROOT = Path(__file__).resolve().parent


def separable(detail, column):
    """Tool x component table of the rms residual, as tool factor * component factor / overall."""
    rms = lambda values: float(np.sqrt(np.mean(values ** 2)))
    overall = rms(detail[column])
    by_tool = detail.groupby('tool')[column].agg(rms)
    by_component = detail.groupby('component')[column].agg(rms).reindex(UDAR_COMPONENTS)
    table = pd.DataFrame(np.outer(by_tool, by_component) / overall, index=by_tool.index, columns=UDAR_COMPONENTS)
    return table, overall


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tables', type=Path, default=ROOT / 'compare-output/model-error')
    parser.add_argument('--points-1d', type=int, nargs=2, default=(69, 102), metavar=('FIRST', 'LAST'))
    parser.add_argument('--points-2d', type=int, nargs='+', default=[45, 60, 80, 95])
    parser.add_argument('--out', type=Path, default=ROOT / 'references/model-error-benchmark3.json')
    args = parser.parse_args(argv)
    one = pd.read_csv(args.tables / 'channels.csv')
    one = one[(one['index'] >= args.points_1d[0]) & (one['index'] <= args.points_1d[1]) & (one['stage'] == '1D')]
    two = pd.read_csv(args.tables / 'channels_2d.csv')
    two = two[two['index'].isin(args.points_2d)]
    table_1d, level_1d = separable(one, 'data_coarse')
    table_2d, level_2d = separable(two, 'truth_2d')
    tools = [tool.split('/') for tool in table_1d.index]
    if list(table_2d.index) != list(table_1d.index):
        raise ValueError('The 1D and 2D tables cover different tools')
    base = {'tools': tools, 'components': list(UDAR_COMPONENTS),
            'rule': 'stage variance = Cd (1 + factor^2); factor = rms mismatch of the true model in units of the '
                    'measurement std, tool factor x component factor',
            'source': {'tables': str(args.tables), 'points_1d': f'{args.points_1d[0]}-{args.points_1d[1]} accepted '
                       f'as 1D ({one["index"].nunique()} points)', 'points_2d': args.points_2d}}
    zero = np.zeros_like(table_1d.values).tolist()
    variants = {'': (table_1d.values.tolist(), table_2d.values.tolist()),
                '-1d': (table_1d.values.tolist(), zero), '-2d': (zero, table_2d.values.tolist())}
    for suffix, (factor_1d, factor_2d) in variants.items():
        path = args.out.with_name(args.out.stem + suffix + '.json')
        path.write_text(json.dumps({**base, 'factor_1d': factor_1d, 'factor_2d': factor_2d}, indent=1) + '\n')
        print(f'wrote {path}')
    for name, table, level in (('1D', table_1d, level_1d), ('2D', table_2d, level_2d)):
        inflation = np.sqrt(1. + table.values ** 2)
        print(f'\n{name} stage: overall rms mismatch {level:.1f} measurement std; the stage std is multiplied by '
              f'{inflation.min():.1f} to {inflation.max():.1f} (median {np.median(inflation):.1f})')
        print(table.round(1).to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
