"""Generate canonical UDAR observations and per-log-point variances."""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from esmda_remote2d import UDAR_COMPONENTS


def read_las(path):
    lines = Path(path).read_text().splitlines()
    names = lines[0].lstrip('%').split()
    missing = set(UDAR_COMPONENTS) - set(names)
    if missing:
        raise ValueError(f'{path}: missing UDAR components {sorted(missing)}')
    columns = [names.index(component) for component in UDAR_COMPONENTS]
    values = np.array([[float(token.replace('D', 'E')) for token in line.split()]
                       for line in lines[1:] if line.strip()], dtype=float)
    return values[:, 0], values[:, columns]


def generate_data(benchmark, tools, output, abs_frac=.025):
    benchmark, output = Path(benchmark), Path(output)
    if not np.isfinite(abs_frac) or abs_frac <= 0:
        raise ValueError('Noise fraction must be positive and finite')
    with h5py.File(benchmark / 'globalmodel.h5', 'r') as handle:
        tvd = handle['wellpath/Z'][:, 0] * 3.28084
        expected_md = handle['wellpath/Distance'][:, 0] * 3.28084
    values = []
    for freq, sub in tools:
        md, datum = read_las(benchmark / 'outputs_2D' / f'UDAR3D_{sub}_{freq}.las')
        if md.shape != expected_md.shape or not np.allclose(md, expected_md, rtol=0, atol=.02):
            raise ValueError(f'LAS trajectory does not match benchmark: {freq}, {sub}')
        if not np.isfinite(datum).all():
            raise ValueError(f'Non-finite observations: {freq}, {sub}')
        values.append(datum)
    values = np.stack(values, axis=1)
    std = np.maximum(abs_frac * np.mean(np.abs(values), axis=1), 1e-12)
    variances = std ** 2
    data = pd.DataFrame({tool: [row.copy() for row in values[:, i]] for i, tool in enumerate(tools)}, index=tvd)
    variance = pd.DataFrame({tool: [['ABS', row.tolist()] for row in variances] for tool in tools}, index=tvd)
    data.index.name = variance.index.name = 'tvd'
    data.attrs.update(components=UDAR_COMPONENTS, benchmark=str(benchmark.resolve()))
    variance.attrs.update(components=UDAR_COMPONENTS, noise_fraction=abs_frac)
    output.mkdir(parents=True, exist_ok=True)
    data.to_pickle(output / 'data.pkl')
    variance.to_pickle(output / 'var.pkl')
    (output / 'assim_index.csv').write_text(''.join(f'{i}\n' for i in range(len(tvd))))
    with (output / 'datatyp.csv').open('w', newline='') as stream:
        csv.writer(stream).writerow([str(tool) for tool in tools])
    return data, variance


def main(argv=None):
    import re
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=here.parent / 'Dashboard/benchmarks/Benchmark-3')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh directory for generated observations')
    parser.add_argument('--noise-fraction', type=float, default=.025)
    args = parser.parse_args(argv)
    tools = []
    for path in sorted((args.benchmark / 'outputs_2D').glob('UDAR3D_*.las')):
        match = re.fullmatch(r'UDAR3D_(\d+)ft_(\d+)kHz', path.stem)
        if match:
            sub, freq = match.groups()
            tools.append((f'{freq}kHz', f'{sub}ft'))
    if not tools:
        raise FileNotFoundError('No 2D benchmark UDAR LAS files found')
    if args.output_dir.exists():
        raise FileExistsError('Use a fresh output directory')
    data, _ = generate_data(args.benchmark, tools, args.output_dir, args.noise_fraction)
    print(f'Wrote {len(data)} rows, {len(tools)} tools, {len(UDAR_COMPONENTS)} channels/tool')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
