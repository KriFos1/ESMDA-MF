"""Drive the gated sequential Matern chain to a target index.

The chain driver runs the local 1D probes until a gate opens and exports the
point. This controller then solves that point's MAP on HPC
(``pilot_rml_index45.py prepare-laplace`` / ``bundle``, one-element sbatch
array, resubmitted if the member stops on its time budget), pulls the result
back, Laplace-resamples it into the chain checkpoint (``chain_laplace.py
sample``) and resumes the chain. It stops once the checkpoint for
``--stop - 1`` exists, or at the next point boundary after a file ``PAUSE``
appears in the chain directory. A restarted controller adopts a MAP job that
is still in flight (or already finished) for the pending point instead of
submitting a second one.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pilot_rml_index45 import TERMINAL_REASONS

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
ACTIVE_STATES = ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED',
                 'RESIZING')


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def run(command, capture=False, env=None):
    log('$ ' + ' '.join(str(part) for part in command))
    completed = subprocess.run([str(part) for part in command], check=True, text=True,
                               capture_output=capture, env=env)
    return completed.stdout if capture else None


def retry(action, attempts=6, delay=60.):
    """Repeat a remote action across transient SSH/network failures."""
    for attempt in range(attempts):
        try:
            return action()
        except subprocess.CalledProcessError as error:
            if attempt + 1 == attempts:
                raise
            log(f'remote command failed ({error}); retrying in {delay:g} s')
            time.sleep(delay)


def ssh(host, command, capture=False):
    return retry(lambda: run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', host, command],
                             capture=capture))


def checkpoint_indices(chain):
    return sorted(int(path.stem.rsplit('_', 1)[1])
                  for path in Path(chain).glob('inversion_results_assim_*.pkl'))


def pending_gate(chain):
    """Index of the newest exported gate point that has no checkpoint yet, or None."""
    done = set(checkpoint_indices(chain))
    gates = sorted(int(path.stem.split('-')[1]) for path in Path(chain).glob('gate-*.json'))
    pending = [index for index in gates if index not in done]
    return pending[-1] if pending else None


def sacct_state(text, job):
    """State of array element 0 of ``job`` from ``sacct -n -P -o JobID,State`` output."""
    for line in text.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 2 and fields[0] == f'{job}_0':
            return fields[1].split()[0]
    return None


def driver_arguments(study, stop):
    anisotropy = study['anisotropy']
    arguments = ['--stop', str(stop), '--ensemble-size', str(study['ensemble_size']),
                 '--seed', str(study['seed']), '--mda-steps', str(study['mda_steps']),
                 '--tempering', study['tempering'], '--anisotropy', anisotropy['parameterization'],
                 *(['--reference', study['reference']] if 'reference' in study else []),
                 '--anisotropy-mean', str(anisotropy['state_moments'][0]),
                 '--anisotropy-std', str(anisotropy['state_moments'][1]), '--resume']
    if study.get('relaxation') not in (None, 'none'):
        arguments += ['--memory-ft', str(study['relaxation']['memory_ft'])]
    if study.get('gate') not in (None, 'none'):
        arguments += ['--gate-threshold', str(study['gate']['threshold']), '--on-gate', 'stop']
    if study.get('analysis', 'esmda') not in ('esmda', None):
        arguments += ['--analysis', study['analysis']['method']]
    if study.get('two_d_prior', 'matern_cellwise') != 'matern_cellwise':
        arguments += ['--two-d-prior', study['two_d_prior']]
    if study.get('directional_sigma_floor'):
        arguments += ['--directional-sigma-floor', *(str(value) for value in study['directional_sigma_floor'])]
    if study.get('likelihood_weight', 1.) != 1.:
        arguments += ['--likelihood-weight', str(study['likelihood_weight'])]
    lateral = study.get('lateral_update', {'mode': 'full'})
    if lateral['mode'] != 'full':
        arguments += ['--lateral-update', lateral['mode'], '--column-halfwidth', str(lateral['column_halfwidth']),
                      '--footprint-ft', str(lateral['footprint_ft']),
                      '--footprint-taper-ft', str(lateral['footprint_taper_ft'])]
    if study.get('cascade') not in (None, 'none'):
        arguments += ['--cascade-threshold', str(study['cascade']['threshold']),
                      '--compare-revert', study['cascade'].get('compare_revert', 'never')]
    return arguments


def job_in_directory(text, workdir):
    """Array job ID from ``squeue -h -o '%A|%Z'`` output whose working directory is ``workdir``."""
    for line in text.splitlines():
        fields = line.strip().split('|')
        if len(fields) == 2 and fields[1].rstrip('/') == workdir.rstrip('/'):
            return fields[0]
    return None


def active_job(host, workdir):
    return job_in_directory(ssh(host, "squeue -u $USER -h -o '%A|%Z'", capture=True), workdir)


def pause_requested(chain):
    if (Path(chain) / 'PAUSE').exists():
        log(f'pause requested ({Path(chain) / "PAUSE"}); stopping at the point boundary')
        return True
    return False


def wait_for_job(host, job, poll):
    while True:
        time.sleep(poll)
        state = sacct_state(ssh(host, f'sacct -j {job} -n -P -o JobID,State', capture=True), job)
        if state is not None and state not in ACTIVE_STATES:
            log(f'job {job}: {state}')
            return state
        log(f'job {job}: {state or "not yet in sacct"}')


def array_states(text, job):
    """{element: state} of array ``job`` from ``sacct -n -P -o JobID,State`` output."""
    states = {}
    for line in text.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 2 and fields[0].startswith(f'{job}_') and fields[0][len(job) + 1:].isdigit():
            states[int(fields[0][len(job) + 1:])] = fields[1].split()[0]
    return states


def wait_for_array(host, job, members, poll):
    """Wait until every listed element of an array job has left the active states."""
    while True:
        time.sleep(poll)
        states = array_states(ssh(host, f'sacct -j {job} -n -P -o JobID,State', capture=True), job)
        # Pending elements appear as one bracketed line; wait until each listed member reports.
        if set(members) <= set(states) and all(states[member] not in ACTIVE_STATES for member in members):
            log(f'job {job}: {states}')
            return states
        log(f'job {job}: {states or "not yet in sacct"}')


NODE_LAUNCHER = r'''#!/bin/bash
# Run the listed MAP members of a prepared study in parallel on a plain node (no Slurm).
# Each member resumes from its checkpoint when stopped by its time budget (exit code 3).
cd "$RML_PROJECT"
export PYTHONPATH="$RML_PROJECT:$RML_PROJECT/pxwf-src" PXWF_SRC="$RML_PROJECT/pxwf-src" RML_NODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
ulimit -s unlimited 2>/dev/null || true
for member in "$@"; do
  (
    for attempt in 1 2 3; do
      "$RML_PYTHON" -B -u pilot_rml_index45.py member --run-dir "$RML_RUN" --member "$member" \
        >> "$RML_RUN/member-$(printf %03d "$member").log" 2>&1
      [ $? -eq 3 ] || break
    done
  ) &
done
wait
rm -f "$RML_RUN/node-running"
'''


def node_running(host, remote):
    """True while a node launcher of this run is alive (its PID in run/node-running)."""
    return ssh(host, f'f={remote}/run/node-running; [ -f $f ] && kill -0 $(cat $f) 2>/dev/null '
                     f'&& echo running || echo idle', capture=True).strip() == 'running'


def node_launch(args, remote, members):
    """Start the members on the node in the background (no Slurm); the PID marks the launch as running."""
    script = f'{remote}/run/node-members.sh'
    ssh(args.host, f"cat > {script} <<'EOF'\n{NODE_LAUNCHER}EOF\nchmod +x {script}")
    # Only the launcher goes to the background (with every stream redirected), so ssh returns at once;
    # backgrounding the whole 'cd && export && nohup' list kept ssh attached until the members finished.
    ssh(args.host, f'cd {remote}; export RML_PROJECT={remote}/code RML_RUN={remote}/run '
                   f'RML_PYTHON={args.hpc_python}; nohup {script} {" ".join(map(str, members))} '
                   f'> {remote}/run/node-launch.log 2>&1 < /dev/null & echo $! > {remote}/run/node-running')


def wait_for_node(args, remote, index):
    while node_running(args.host, remote):
        time.sleep(args.poll)
    log(f'index {index}: node launcher finished')


def select_member(run_dir, count):
    """Lowest final objective among the finished members (same prior and data, so comparable)."""
    objectives = {}
    for member in range(count):
        report, laplace = run_dir / f'member-{member:03d}' / 'result.json', run_dir / f'member-{member:03d}' / 'laplace.npz'
        if report.exists() and laplace.exists():
            result = json.loads(report.read_text())
            if result.get('reason') in TERMINAL_REASONS and result.get('history'):
                objectives[member] = float(result['history'][-1]['objective'])
    if not objectives:
        return None
    best = min(objectives, key=objectives.get)
    (run_dir / 'selected-member.json').write_text(json.dumps({'member': best, 'objectives': objectives}, indent=1))
    return best


def map_stage(args, chain, index, problem=None, suffix='', compare=False):
    """MAP on HPC for an exported problem, then the Laplace resampling.

    By default the gate point's own problem, sampled into the chain; with
    ``compare`` an alternative problem (e.g. the reverted cascade point),
    resampled into ``<run>/compare`` without touching the chain. With
    ``--map-starts K`` the MAP runs K + 1 starts (prior mean and K carried
    members) and the lowest objective is resampled (``selected-member.json``).
    """
    problem = problem or f'laplace-problem-{index:03d}.npz'
    run_dir = ROOT / 'runs' / f'chain-laplace-idx{index:03d}-{args.label}{suffix}'
    remote = f'{args.hpc_root}/chain-laplace-idx{index:03d}-{args.label}{suffix}'
    starts = getattr(args, 'map_starts', 0)
    node = getattr(args, 'executor', 'slurm') == 'node'

    def pull():
        retry(lambda: run(['rsync', '-a', '--exclude', 'jobs/', '--exclude', 'deployment.tar.gz',
                           '--exclude', 'inputs.npz', '--exclude', 'study.json',
                           '--exclude', 'prior-moments.npz', f'{args.host}:{remote}/run/', f'{run_dir}/']))

    def result_of(member):
        report = run_dir / f'member-{member:03d}' / 'result.json'
        return json.loads(report.read_text()) if report.exists() else {}

    def finished(member):
        return (result_of(member).get('reason') in TERMINAL_REASONS
                and (run_dir / f'member-{member:03d}' / 'laplace.npz').exists())

    fresh = not run_dir.exists()
    if fresh:
        overrides = [*(['--binary-dir', args.binary_dir] if args.binary_dir else []),
                     *(['--orientation', args.orientation] if args.orientation else []),
                     *(['--starts', str(starts), '--start-seed', str(args.seed_base + index), '--include-mean-start']
                       if starts else []),
                     *(['--node-solver', str(args.node_threads), str(args.node_concurrency)] if node else []),
                     *(item for setting in (getattr(args, 'optimizer_setting', None) or [])
                       for item in ('--optimizer-setting', setting)),
                     *(['--hdfql', args.node_hdfql] if node and args.node_hdfql else [])]
        run([PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'prepare-laplace', '--problem',
             chain / problem, '--source', args.source, '--run-dir', run_dir, *overrides])
        run([PYTHON, '-B', ROOT / 'pilot_rml_index45.py', 'bundle', '--run-dir', run_dir])
        ssh(args.host, f'mkdir -p {remote}')
        retry(lambda: run(['scp', '-q', run_dir / 'deployment.tar.gz', f'{args.host}:{remote}/']))
        ssh(args.host, f'cd {remote} && tar xzf deployment.tar.gz && rm deployment.tar.gz')
    count = json.loads((run_dir / 'study.json').read_text())['count']
    if not all(finished(member) for member in range(count)):
        # A restarted controller first collects results or a job left by its predecessor.
        if not fresh:
            pull()
        if node:
            in_flight = None if fresh else ('node' if node_running(args.host, remote) else None)
        else:
            in_flight = None if fresh else active_job(args.host, f'{remote}/run')
        for submission in range(args.max_submissions):
            pending = [member for member in range(count) if not finished(member)]
            if not pending:
                break
            if in_flight is not None:
                job, in_flight = in_flight, None
                log(f'index {index}: adopting MAP job {job} already in flight')
            elif node:
                node_launch(args, remote, pending)
                job = 'node'
                log(f'index {index}: MAP members {pending} started on {args.host} (submission {submission + 1})')
            else:
                job = ssh(args.host, f'cd {remote} && export RML_PROJECT={remote}/code RML_RUN={remote}/run '
                                     f'RML_PYTHON={args.hpc_python} && sbatch --parsable --export=ALL '
                                     f'--array={",".join(map(str, pending))} --chdir=$RML_RUN '
                                     f'$RML_RUN/rml-array.sbatch', capture=True).strip()
                log(f'index {index}: MAP job {job} submitted for members {pending} (submission {submission + 1})')
            if node:
                wait_for_node(args, remote, index)
            else:
                wait_for_array(args.host, job, pending, args.poll)
            pull()
            for member in pending:
                result = result_of(member)
                log(f"index {index}: member {member} reason {result.get('reason')}, "
                    f"{result.get('accepted_iterations')} iterations, {result.get('seconds_this_attempt')} s")
                if result.get('reason') == 'execution_failure':
                    # The member reruns from its checkpoint on resubmission; solver crashes are
                    # usually transient (node, I/O), persistent ones end in the fallback below.
                    log(f"index {index}: member {member} execution failure ({str(result.get('error'))[:200]})")
    best = select_member(run_dir, count)
    if best is None:
        raise RuntimeError(f'index {index}: no MAP member finished within {args.max_submissions} submissions')
    if count > 1:
        objectives = json.loads((run_dir / 'selected-member.json').read_text())['objectives']
        log(f'index {index}: objectives {dict((int(k), round(v, 1)) for k, v in objectives.items())}; '
            f'member {best} selected')
    if compare:
        run([PYTHON, '-B', ROOT / 'chain_laplace.py', 'compare', '--chain', chain, '--problem', chain / problem,
             '--map', run_dir, '--output', run_dir / 'compare', '--count', str(args.count),
             '--seed', str(args.seed_base + index)])
    else:
        run([PYTHON, '-B', ROOT / 'chain_laplace.py', 'sample', '--chain', chain, '--map', run_dir,
             '--count', str(args.count), '--seed', str(args.seed_base + index)])
    # Solver scratch (tens of GB per point) is not needed once the result is local.
    ssh(args.host, f'rm -rf {remote}/run/member-*/jobs {remote}/run/member-*/checkpoint.npz')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', type=Path, required=True)
    parser.add_argument('--stop', type=int, required=True, help='Exclusive end index')
    parser.add_argument('--source', type=Path, default=ROOT / 'runs/rml-index45-lm-20261019b')
    parser.add_argument('--binary-dir', default=None, help='fwd2D build for the MAP stages (default: source study)')
    parser.add_argument('--orientation', choices=['gradient', 'bedding'], default=None,
                        help='Material orientation for the MAP stages (default: source study)')
    parser.add_argument('--host', default='HPC')
    parser.add_argument('--hpc-root', default='/cluster/home/krfo/rml-index45-20261019')
    parser.add_argument('--hpc-python', default='/cluster/home/krfo/rml-index45-20261019/.venv/bin/python')
    parser.add_argument('--label', default=time.strftime('%Y%m%d'))
    parser.add_argument('--count', type=int, default=250)
    parser.add_argument('--seed-base', type=int, default=7000)
    parser.add_argument('--poll', type=float, default=30.)
    parser.add_argument('--max-submissions', type=int, default=3)
    parser.add_argument('--optimizer-setting', action='append', default=None, metavar='KEY=VALUE',
                        help='Override one MAP optimizer setting, e.g. stagnation_tolerance=0.01 (repeatable)')
    parser.add_argument('--executor', choices=['slurm', 'node'], default='slurm',
                        help='slurm: sbatch arrays on --host; node: background processes on a plain node')
    parser.add_argument('--node-threads', type=int, default=3, help='node: threads per single-rank solve')
    parser.add_argument('--node-concurrency', type=int, default=4, help='node: candidate solves at once per member')
    parser.add_argument('--node-hdfql', default=None, help='node: directory with libHDFql.so')
    parser.add_argument('--map-starts', type=int, default=0,
                        help='Extra MAP starts from carried members (besides the prior mean); the lowest '
                             'objective is resampled')
    parser.add_argument('--max-consecutive-fallbacks', type=int, default=3,
                        help='Gate points carried without a 2D stage after a failed MAP before aborting')
    args = parser.parse_args(argv)
    chain = args.chain.resolve()
    study = json.loads((chain / 'study.json').read_text())
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    consecutive_fallbacks = 0
    while True:
        done = checkpoint_indices(chain)
        if done and done[-1] >= args.stop - 1:
            log(f'chain complete through index {done[-1]}')
            return 0
        if pause_requested(chain):
            return 0
        gate = pending_gate(chain)
        if gate is None:
            run([PYTHON, '-B', '-u', ROOT / 'sequential_matern_1d.py', '--output-dir', chain,
                 *driver_arguments(study, args.stop)], env=env)
            gate = pending_gate(chain)
            if gate is None:
                continue
            if pause_requested(chain):   # the exported gate point stays pending for the next start
                return 0
        log(f'index {gate}: gate open, running the Laplace 2D stage')
        record = json.loads((chain / f'gate-{gate:03d}.json').read_text())
        side = ROOT / 'runs' / f'chain-laplace-idx{gate:03d}-{args.label}-revert' / 'compare' / 'summary.json'
        if record.get('revert_problem') and not side.exists():
            # Side-by-side check (first cascade gate): the same point with the 1D update reverted.
            try:
                log(f"index {gate}: side-by-side MAP for {record['revert_problem']}")
                map_stage(args, chain, gate, problem=record['revert_problem'], suffix='-revert', compare=True)
            except Exception as error:   # the comparison must not stop the chain
                log(f'index {gate}: side-by-side stage failed ({error}); continuing with the chain')
        try:
            map_stage(args, chain, gate)
            consecutive_fallbacks = 0
        except Exception as error:   # keep an unattended chain moving; the fallback is recorded
            consecutive_fallbacks += 1
            log(f'index {gate}: Laplace stage failed ({error}); carrying the reverted prior '
                f'(fallback {consecutive_fallbacks}/{args.max_consecutive_fallbacks})')
            if consecutive_fallbacks > args.max_consecutive_fallbacks:
                raise
            run([PYTHON, '-B', ROOT / 'chain_laplace.py', 'carry', '--chain', chain,
                 '--reason', str(error)[:300]])


if __name__ == '__main__':
    raise SystemExit(main())
