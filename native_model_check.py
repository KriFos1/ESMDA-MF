"""Can our fwd2D builds reproduce the Benchmark-3 data from the benchmark's own model?

Every build runs on the native ``Benchmark-3/globalmodel.h5`` (no window, no
upscaling) at single logging points, with the benchmark's ``tool.inp``,
datflag 3 (its 8 components are the first 8 of the data's datflag-5 output,
same formulas), ``depthref = 0`` (tool midpoint) and the original selection
tolerance 1e-3, with and without the original distribution's
``customgrids.inp`` (20 4 40 / 30 20 30 / 2 1 2; without it the solver
defaults to 12 10 16 optimal nodes). ``prepare`` writes the HPC job; ``compare``
scores every run against ``outputs_2D`` (the chain's data) with the chain's
mask and Cd, phase residuals wrapped.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT.parent / 'Dashboard/benchmarks/Benchmark-3'
HOME = '/cluster/home/krfo'
BUILDS = {  # name: (binary, orientation flag or None)
    'original-v4.2': (f'{HOME}/LogSimulator3D_v4.2fullfortran/fwd2D', None),
    'jacobi-nrx': (f'{HOME}/LogSimulator3D_v4.2fullfortran_jacobi_nrx_2D_3D/fwd2D', None),
    'adjoint-v1': (f'{HOME}/adjoint-2d-v1/fwd2D', None),
    'orient-gradient': (f'{HOME}/adjoint-2d-v1-orient/fwd2D', None),
    'orient-bedding': (f'{HOME}/adjoint-2d-v1-orient/fwd2D', 0),
}
GRIDS = {'default-grid': None, 'customgrids': '20 4 40\n30 20 30\n2 1 2\n'}
POINTS = [10, 33, 40, 44, 130]
TOLERANCE = 1e-3
CHAIN_SETTINGS = ('orient-bedding', 'default-grid', 1e-6)   # what the chain's MAP actually ran


def cases():
    for build in BUILDS:
        for grid in GRIDS:
            for point in POINTS:
                yield build, grid, TOLERANCE, point
    build, grid, tolerance = CHAIN_SETTINGS
    for point in POINTS:
        yield build, grid, tolerance, point


def case_name(build, grid, tolerance, point):
    return f'{build}__{grid}__tol{tolerance:g}__idx{point:03d}'


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / 'inputs').mkdir()
    (output / 'inputs/tool.inp').write_text((BENCHMARK / 'tool.inp').read_text())
    (output / 'inputs/outputsettings.inp').write_text('3\n1\n0\n1e-10\n')
    lines = []
    for build, grid, tolerance, point in cases():
        binary, orientation = BUILDS[build]
        name = case_name(build, grid, tolerance, point)
        setup = [f'mkdir -p {name}/outputs', f'cd {name}', 'ln -sf ../globalmodel.h5 globalmodel.h5',
                 'cp ../inputs/tool.inp ../inputs/outputsettings.inp .',
                 f"printf '{point + 1}\\n{point + 1}\\n0\\n{tolerance:g}\\n' > selection.inp"]
        if GRIDS[grid]:
            setup.append(f"printf '{GRIDS[grid].replace(chr(10), chr(92) + 'n')}' > customgrids.inp")
        if orientation is not None:
            setup.append(f"printf '{orientation}\\n' > orientation.inp")
        setup += [f'echo {name}; sha256sum {binary} > binary.sha256',
                  f'srun --mpi=pmi2 --nodes=1 --ntasks=2 --cpus-per-task=16 {binary} > solver.log 2>&1 '
                  f'|| echo "FAILED {name}"', 'cd ..']
        lines.append('( ' + ' && '.join(setup[:-1]) + ' ); ')
    script = f'''#!/bin/bash
#SBATCH --job-name=native-model-check
#SBATCH --partition=comp
#SBATCH --account=energytech
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=native-%j.log
cd "$SLURM_SUBMIT_DIR"
source $(ls -d {HOME}/intel/oneapi/oneapi-vars.sh {HOME}/intel/oneapi/*/oneapi-vars.sh 2>/dev/null | head -1) >/dev/null 2>&1
export LD_LIBRARY_PATH={HOME}/LogSimulator3D_v4.2fullfortran_jacobi_nrx_2D_3D/src/hdfql_linux:${{LD_LIBRARY_PATH:-}}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_STACKSIZE=64M
ulimit -s unlimited
''' + '\n'.join(lines) + '\necho done\n'
    (output / 'native.sbatch').write_text(script)
    (output / 'cases.json').write_text(json.dumps([dict(zip(('build', 'grid', 'tolerance', 'point'), case))
                                                   | {'name': case_name(*case)} for case in cases()], indent=1))
    print(output)


def read_prediction(directory, tools):
    """The 8 UDAR components per tool from a run's outputs/*_<sub>ft_<freq>kHz.las, tool order of the data."""
    values = []
    for freq, sub in tools:
        paths = list((directory / 'outputs').glob(f'UDAR_{sub}_{freq}.las'))
        if len(paths) != 1:
            return None
        rows = [line.split() for line in paths[0].read_text().splitlines()
                if line.strip() and not line.lstrip().startswith(('%', '#', '~'))]
        values.append([float(token.replace('D', 'E')) for token in rows[0][1:9]])
    return np.concatenate(values)


def compare(args):
    import pandas as pd
    import run_DA_sequential as sequential
    from write_data import read_las

    data, variances = pd.read_pickle(args.data), pd.read_pickle(args.variance)
    tools = list(data.columns)
    rows = []
    for case in json.loads((args.output / 'cases.json').read_text()):
        observations, covariance = sequential.observation_vectors(data, variances, case['point'], tools)
        sub_cell = np.zeros(len(observations), dtype=bool)
        sub_cell[sequential.component_positions(len(observations))] = True
        mask = ~(sequential.noise_floor_mask(observations, covariance) | sub_cell)
        prediction = read_prediction(args.output / case['name'], tools)
        row = dict(case)
        if prediction is None:
            row['status'] = 'missing'
        else:
            residual = prediction - observations
            residual[::2] = (residual[::2] + 180.) % 360. - 180.     # phases are the even components
            row['chi2_per_datum'] = float(np.mean(residual[mask] ** 2 / covariance[mask]))
            row['max_abs_sigma'] = float(np.max(np.abs(residual[mask]) / np.sqrt(covariance[mask])))
            log = (args.output / case['name'] / 'solver.log').read_text(errors='replace')
            row['ifdip_logged'] = sorted(set(re.findall(r'ifdip\s*=\s*(-?\d+)', log)))
            row['grid_logged'] = sorted(set(re.findall(r'NZ_ext\s*=\s*(\d+)', log)))
        rows.append(row)
    (args.output / 'comparison.json').write_text(json.dumps(rows, indent=1))
    table = pd.DataFrame(rows)
    if 'chi2_per_datum' in table:
        print(table.pivot_table(index=['build', 'grid', 'tolerance'], columns='point', values='chi2_per_datum')
              .round(3).to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'compare'):
        command = sub.add_parser(name)
        command.add_argument('--output', type=Path, required=True)
        if name == 'compare':
            command.add_argument('--data', type=Path, default=ROOT / 'runs/benchmark3-data-20261002/data.pkl')
            command.add_argument('--variance', type=Path, default=ROOT / 'runs/benchmark3-data-20261002/var.pkl')
    args = parser.parse_args(argv)
    {'prepare': prepare, 'compare': compare}[args.command](args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
