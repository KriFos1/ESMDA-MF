"""Start or resume the IA field-data multi-fidelity chain on HPC."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def implementation_id():
    """Identify the code bundled into new MAP stages, including the worker."""
    from pilot_rml_index45 import MODULES

    paths = [ROOT / name for name in MODULES]
    paths += [ROOT.parent / 'Dashboard/src/pxwf/execution' / name
              for name in ('worker.py', 'backend.py', 'config.py', 'adjoint.py')]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=None)
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--lwd-prior', choices=['section-global', 'pointwise'], default='section-global')
    parser.add_argument('--stop', type=int, default=379)
    parser.add_argument('--ensemble-size', type=int, default=250)
    parser.add_argument('--mda-steps', type=int, default=10)
    parser.add_argument('--gate-threshold', type=float, default=25.)
    parser.add_argument('--memory-ft', type=float, default=50.)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument('--host', default='HPC')
    parser.add_argument('--binary-dir', default='/cluster/home/krfo/ia-dip-build-20261030/solver')
    parser.add_argument('--hpc-root', default='/cluster/home/krfo/rml-index45-20261019')
    parser.add_argument('--hpc-python', default='/cluster/home/krfo/rml-index45-20261019/.venv/bin/python')
    parser.add_argument('--label', default=None)
    parser.add_argument('--map-starts', type=int, default=0)
    parser.add_argument('--fixed-zero-dip', action='store_true', help='Fix apparent dip at zero throughout the chain')
    parser.add_argument('--solver-grid', type=int, nargs=9, default=None,
                        metavar=('UX', 'UY', 'UZ', 'OX', 'OY', 'OZ', 'AX', 'AY', 'AZ'),
                        help="2D solver grid for the MAP stages, e.g. 40 4 80 30 20 30 2 1 2 "
                             "(default: the solver's built-in grid)")
    parser.add_argument('--solver-tolerance', type=float, default=None,
                        help='2D solver tolerance for the MAP stages (default: 1e-6)')
    parser.add_argument('--evaluation-timeout', type=float, default=None,
                        help='Seconds one MAP-stage solver evaluation may run (default: 180; refined '
                             'grids need more)')
    parser.add_argument('--node-sbatch', type=int, nargs=2, default=None, metavar=('THREADS', 'CONCURRENCY'),
                        help='MAP stages with single-rank node solves inside the job (e.g. 8 4) instead of '
                             '2-rank srun steps, which deadlock on a convergence error')
    parser.add_argument('--first-map-seconds', type=float, default=None,
                        help="Optimizer budget of the chain's first (cold-start) MAP")
    parser.add_argument('--optimizer-setting', action='append', default=None, metavar='KEY=VALUE',
                        help='MAP optimizer override, e.g. max_seconds=120 or initial_trust=2.0 (repeatable)')
    parser.add_argument('--rh-ranges', type=float, nargs=2, default=None, metavar=('X_FT', 'Z_FT'),
                        help='Prior ln Rh Matern ranges (new chains; default 50 50)')
    parser.add_argument('--latent-ranges', type=float, nargs=2, default=None, metavar=('X_FT', 'Z_FT'),
                        help='Prior anisotropy Matern ranges (new chains; default 80 35)')
    args = parser.parse_args(argv)
    if args.map_starts < 0:
        parser.error('--map-starts must be nonnegative')
    run_dir = args.run_dir.resolve()
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    if not run_dir.exists():
        command = [sys.executable, '-B', '-u', str(ROOT / 'run_ia_chain.py'), '--output-dir', str(run_dir),
                   '--stop', str(args.stop), '--ensemble-size', str(args.ensemble_size),
                   '--mda-steps', str(args.mda_steps), '--gate-threshold', str(args.gate_threshold),
                   '--memory-ft', str(args.memory_ft), '--seed', str(args.seed), '--lwd-prior', args.lwd_prior]
        if args.inputs is not None:
            command += ['--inputs', str(args.inputs.resolve())]
        if args.data_dir is not None:
            command += ['--data-dir', str(args.data_dir.resolve())]
        if args.fixed_zero_dip:
            command += ['--fixed-zero-dip']
        if args.rh_ranges is not None:
            command += ['--rh-ranges', *map(str, args.rh_ranges)]
        if args.latent_ranges is not None:
            command += ['--latent-ranges', *map(str, args.latent_ranges)]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    study = json.loads((run_dir / 'study.json').read_text())
    if study.get('method') != 'ia_field_dip_enif':
        raise ValueError('Existing directory is not an IA field-dip chain')
    if study.get('fixed_zero_dip', False) != args.fixed_zero_dip:
        raise ValueError('Existing chain uses a different dip mode')
    if study.get('lwd_prior_mode', 'pointwise') != args.lwd_prior:
        raise ValueError('Existing chain uses a different LWD prior; use a new run directory for the global prior')
    for name in ('inputs', 'data_dir'):
        supplied = getattr(args, name)
        if supplied is not None and supplied.resolve() != Path(study[name]):
            raise ValueError(f'Existing chain uses a different {name}')
    binary_id = hashlib.sha256(args.binary_dir.encode()).hexdigest()[:8]
    solver = [args.solver_grid, args.solver_tolerance, args.evaluation_timeout, args.node_sbatch,
              args.first_map_seconds, args.optimizer_setting]
    solver_id = ('' if solver == [None] * 6 else
                 '-s' + hashlib.sha256(json.dumps(solver).encode()).hexdigest()[:6])
    label = args.label or f'ia-{run_dir.name}-{implementation_id()}-b{binary_id}-m{args.map_starts}{solver_id}'
    if not label or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for character in label):
        parser.error('Label must contain only letters, digits, hyphen or underscore')
    command = [sys.executable, '-B', '-u', str(ROOT / 'run_chain_laplace_loop.py'),
               '--chain', str(run_dir), '--stop', str(args.stop), '--count', str(study['ensemble_size']),
               '--source', str(ROOT / 'references/map-source'), '--label', label,
               '--map-starts', str(args.map_starts), '--binary-dir', args.binary_dir,
               '--orientation', 'bedding', '--host', args.host, '--hpc-root', args.hpc_root,
               '--hpc-python', args.hpc_python]
    if args.solver_grid is not None:
        command += ['--solver-grid', *map(str, args.solver_grid)]
    if args.solver_tolerance is not None:
        command += ['--solver-tolerance', str(args.solver_tolerance)]
    if args.evaluation_timeout is not None:
        command += ['--evaluation-timeout', str(args.evaluation_timeout)]
    if args.node_sbatch is not None:
        command += ['--node-sbatch', *map(str, args.node_sbatch)]
    if args.first_map_seconds is not None:
        command += ['--first-map-seconds', str(args.first_map_seconds)]
    for setting in args.optimizer_setting or []:
        command += ['--optimizer-setting', setting]
    print(f'{time.strftime("%H:%M:%S")} starting IA controller for {run_dir}', flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
