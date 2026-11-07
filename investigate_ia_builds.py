"""Re-run saved IA solver evaluations with every fwd2D build, several times each.

Each case is a saved worker job directory (request.json + inputs.h5: the exact model, tool
position, dip and settings the chain evaluated). Each build evaluates the identical request
on the same node, ``--repeats`` times, so the table separates three things: a build that
fails where an older one converges (a regression introduced by our modifications), a model
that fails in every build (the solver itself), and a run-to-run difference on identical
inputs (non-determinism). Builds without an adjoint spec only get the forward cases.
Single-rank requests run the binary directly, as the chain's node solves do.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from pxwf.execution import worker


def build_runtime(runtime, directory):
    directory = Path(directory).resolve()
    binary = directory / 'fwd2D'
    stat = binary.stat()
    hdfql = directory if (directory / 'libHDFql.so').exists() else directory / 'src' / 'hdfql_linux'
    spec = directory / 'adjoint-spec.json'
    out = {**runtime, 'binary': str(binary), 'hdfql': str(hdfql), 'binary_size': stat.st_size,
           'binary_mtime_ns': stat.st_mtime_ns}
    if spec.exists():
        out.update(adjoint_spec=str(spec), adjoint_spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest())
    else:
        out.pop('adjoint_spec', None)
        out.pop('adjoint_spec_sha256', None)
    return out


def adapted_metadata(metadata, runtime):
    """Drop what a build cannot honour (fixed orientation, apparent dip); the build then uses its own rule.

    Returns the metadata and the list of stripped keys, so a comparison across builds states
    when the physics differs (TI axis from the build's gradient rule, no apparent dip).
    """
    spec = json.loads(Path(runtime['adjoint_spec']).read_text()) if runtime.get('adjoint_spec') else {}
    metadata, stripped = dict(metadata), []
    if spec.get('orientation_input') != worker.ORIENTATION_INPUT:
        for key in ('orientation', 'apparent_dip_degrees'):
            if key in metadata:
                stripped.append(key)
                del metadata[key]
    elif 'apparent_dip_degrees' in metadata and spec.get('apparent_dip_input') != worker.DIP_INPUT:
        stripped.append('apparent_dip_degrees')
        del metadata['apparent_dip_degrees']
    return metadata, stripped


def evaluate(work, request, fields, runtime):
    metadata, resources = request['metadata'], dict(request['resources'])
    metadata, stripped = adapted_metadata(metadata, runtime)
    slurm = resources['ranks'] > 1
    if slurm:
        resources['srun_options'] = [*resources.get('srun_options', []), '--kill-on-bad-exit=1']
    started = time.monotonic()
    try:
        outcome = worker.evaluate(work, metadata, fields, runtime, resources, slurm=slurm,
                                  profile=request['profile'])
        result = {'state': 'READY', 'prediction': np.asarray(outcome[0], dtype=float).ravel().tolist()}
    except Exception as error:
        result = {'state': 'FAILED', 'error': f'{type(error).__name__}: {error}'[:300]}
    result['seconds'] = time.monotonic() - started
    result['stripped_metadata'] = stripped
    log = work / 'solver.log'
    text = log.read_text(errors='replace').replace('\x00', '') if log.exists() else ''
    bounds = [float(v) for v in __import__('re').findall(r'Error Bound:\s*([0-9.E+-]+)', text)]
    result.update(rom_exits=sorted({line.strip()[:40] for line in text.splitlines() if 'EXIT ROM' in line}),
                  sigsegv='SIGSEGV' in text, converged_lines=text.count('Solver converged'),
                  min_error_bound=min(bounds) if bounds else None, bound_lines=len(bounds))
    return result


def wrapped_difference(a, b):
    difference = np.subtract(a, b, dtype=float)
    difference[0::2] = (difference[0::2] + 180.) % 360. - 180.
    return difference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', default=[], metavar='NAME=JOB_DIR')
    parser.add_argument('--build', action='append', default=[], metavar='NAME=DIR')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--solver-tolerance', type=float, action='append', default=None,
                        help='Evaluate at these tolerances instead of the saved one (repeatable)')
    parser.add_argument('--threads', type=int, action='append', default=None,
                        help='Solver threads (OMP and MKL) instead of the saved count (repeatable; variants '
                             'are named build@tol/threads)')
    parser.add_argument('--oneapi', default=None,
                        help='oneapi-vars.sh to source instead of the one saved in the requests (another machine)')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    builds = [item.split('=', 1) for item in args.build]
    report = {'repeats': args.repeats, 'builds': dict(builds), 'cases': []}
    for item in args.case:
        name, source = item.split('=', 1)
        source = Path(source).resolve()
        request = json.loads((source / 'request.json').read_text())
        with h5py.File(source / 'inputs.h5') as inputs:
            fields = inputs['fields'][0]
        case = {'name': name, 'source': str(source), 'jacobian': bool(request['metadata'].get('jacobian')),
                'dip_degrees': request['metadata'].get('apparent_dip_degrees'),
                'solver_tolerance': request['metadata'].get('solver_tolerance'), 'runs': {}}
        tolerances = args.solver_tolerance or [request['metadata'].get('solver_tolerance')]
        saved_threads = request['resources'].get('cpus_per_rank')
        for build, directory in builds:
          for tolerance in tolerances:
           for threads in (args.threads or [saved_threads]):
            runtime = build_runtime(request['runtime'], directory)
            if args.oneapi:
                runtime['oneapi'] = str(Path(args.oneapi).expanduser().resolve())
            if case['jacobian'] and 'adjoint_spec' not in runtime:
                continue
            variant = build if tolerance == request['metadata'].get('solver_tolerance') else f'{build}@{tolerance:g}'
            if threads != saved_threads:
                variant += f'/{threads}'
            request_t = {**request, 'metadata': {**request['metadata'], 'solver_tolerance': tolerance},
                         'resources': {**request['resources'], 'cpus_per_rank': threads, 'mkl_threads': threads}}
            runs = []
            for repeat in range(args.repeats):
                result = evaluate(output / name / variant / f'run-{repeat}', request_t, fields, runtime)
                runs.append(result)
                brief = {k: v for k, v in result.items() if k != 'prediction'}
                print(json.dumps({'case': name, 'build': variant, 'repeat': repeat, **brief}), flush=True)
            ready = [r['prediction'] for r in runs if r['state'] == 'READY']
            case['runs'][variant] = {'states': [r['state'] for r in runs], 'seconds': [round(r['seconds'], 1) for r in runs],
                                   'min_error_bound': [r['min_error_bound'] for r in runs],
                                   'rom_exits': [r['rom_exits'] for r in runs],
                                   'repeat_spread': (float(np.abs(wrapped_difference(ready[0], ready[-1])).max())
                                                     if len(ready) > 1 else None),
                                   'prediction': ready[0] if ready else None}
        report['cases'].append(case)
        (output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
