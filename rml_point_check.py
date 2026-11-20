#!/usr/bin/env python3
"""RML check of the 2D Laplace step at one gated point of a finished chain.

Solves --members randomized MAP problems on the HPC (own prior draw and perturbed data each, same prior and
data covariance as the chain's MAP at that point) and pulls the results to runs/point-tests/rml/rml-idx<index>-<label>.
The chain itself is not touched. Compare with the Laplace posterior afterwards (rml_point_compare.py).

python -B -u rml_point_check.py --chain runs/fault-mean-g25-s100 --index 115 --label fault-mean
"""
import argparse
import json
from pathlib import Path

import run_chain_laplace_loop as loop

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--members', type=int, default=24)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=3, help='Array elements (one node each) running at once')
    parser.add_argument('--source', type=Path, default=ROOT / 'references/map-source')
    parser.add_argument('--binary-dir', default='/cluster/home/mlie/cluster-build')
    parser.add_argument('--orientation', default='bedding')
    parser.add_argument('--host', default='HPC')
    parser.add_argument('--hpc-root', default='/cluster/work/users/mlie/esmda-runs')
    parser.add_argument('--hpc-python', default='/cluster/home/mlie/esmda-runs/.venv/bin/python')
    parser.add_argument('--poll', type=float, default=60.)
    parser.add_argument('--optimizer-setting', action='append', metavar='KEY=VALUE',
                        default=['stagnation_tolerance=0.01', 'stagnation_iterations=1', 'max_rounds=2'])
    args = parser.parse_args(argv)
    name = f'rml-idx{args.index:03d}-{args.label}'
    run_dir, remote = ROOT / 'runs' / 'point-tests' / 'rml' / name, f'{args.hpc_root}/{name}'
    problem = args.chain.resolve() / f'laplace-problem-{args.index:03d}.npz'

    def pull():
        loop.retry(lambda: loop.run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'deployment.tar.gz',
                                     '--exclude', 'checkpoint.npz', '--exclude', 'inputs.npz', '--exclude',
                                     'study.json', '--exclude', 'prior-precision.npz',
                                     f'{args.host}:{remote}/run/', f'{run_dir}/']))

    def finished(member):
        report = run_dir / f'member-{member:03d}' / 'result.json'
        return (report.exists() and json.loads(report.read_text()).get('reason') in loop.TERMINAL_REASONS
                and (run_dir / f'member-{member:03d}' / 'fit.npz').exists())

    if not run_dir.exists():
        loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'prepare-laplace', '--problem', problem,
                  '--source', args.source, '--run-dir', run_dir, '--binary-dir', args.binary_dir,
                  '--orientation', args.orientation, '--rml', str(args.members), '--rml-seed', str(args.seed),
                  *(item for setting in args.optimizer_setting for item in ('--optimizer-setting', setting))])
        loop.run([loop.PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'bundle', '--run-dir', run_dir])
        loop.ssh(args.host, f'mkdir -p {remote}')
        loop.retry(lambda: loop.run(['scp', '-q', run_dir / 'deployment.tar.gz', f'{args.host}:{remote}/']))
        loop.ssh(args.host, f'cd {remote} && tar xzf deployment.tar.gz && rm deployment.tar.gz')
    else:
        pull()
    for submission in range(3):
        pending = [member for member in range(args.members) if not finished(member)]
        if not pending:
            break
        job = loop.active_job(args.host, f'{remote}/run')
        if job is None:
            job = loop.ssh(args.host, f'cd {remote} && export RML_PROJECT={remote}/code RML_RUN={remote}/run '
                                      f'RML_PYTHON={args.hpc_python} && sbatch --parsable --export=ALL '
                                      f'--array={",".join(map(str, pending))}%{args.concurrency} '
                                      f'--chdir=$RML_RUN $RML_RUN/rml-array.sbatch', capture=True).strip()
            loop.log(f'RML job {job} submitted for {len(pending)} members (submission {submission + 1})')
        loop.wait_for_array(args.host, job, pending, args.poll)
        pull()
    done = [member for member in range(args.members) if finished(member)]
    loop.log(f'{len(done)} of {args.members} RML members finished: {run_dir}')
    # Solver scratch is not needed once the results are local.
    loop.ssh(args.host, f'rm -rf {remote}/run/member-*/jobs {remote}/run/member-*/checkpoint.npz')
    return 0 if len(done) == args.members else 1


if __name__ == '__main__':
    raise SystemExit(main())
