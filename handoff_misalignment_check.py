"""1D-to-2D handoff check: the best member of a 1D MDA point, evaluated in 2D.

Reads a checkpoint written by ``sequential_matern_1d.py`` and selects the member
with the lowest final 1D misfit (original Cd). Two fields go to the 2D
simulator in one batch:

- its full window field;
- the x-invariant extension of its bit column (the layered medium the 1D
  simulator sees).

Predicted phases are moved to the branch nearest the observations. The report
separates:

- the simulator misalignment: 2D on the x-invariant field against 1D, the same
  layered medium;
- the lateral-structure effect: 2D on the full field against 2D on the
  x-invariant field.

``--include-prior`` adds the same member's prior field (before the 1D MDA) and
its x-invariant copy, so the effect of the 1D update on the 2D fit is measured
on one model.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle

import numpy as np

from esmda_core import LocalWindow, align_phases, phase_channels
from esmda_remote2d import UDAR_COMPONENTS, SimBackendConfig, make_remote2d
import run_DA_sequential as sequential

COMPARISONS = {'simulator_misalignment': ('2d_invariant', '1d'),
               'lateral_structure': ('2d_full', '2d_invariant'),
               'handoff_total': ('2d_full', '1d')}
PRIOR_COMPARISONS = {'prior_simulator_misalignment': ('2d_prior_invariant', '1d_prior'),
                     'prior_lateral_structure': ('2d_prior_full', '2d_prior_invariant'),
                     'update_effect_2d_full': ('2d_full', '2d_prior_full')}


def best_member(misfits):
    misfits = np.asarray(misfits, dtype=float)
    if not np.isfinite(misfits).any():
        raise ValueError('No member has a finite misfit')
    return int(np.nanargmin(misfits))


def invariant_extension(field, nx, nz):
    """The bit column of both blocks (ln Rh, anisotropy latent) repeated across x."""
    blocks = np.asarray(field, dtype=float).reshape(2, nx, nz)
    bit = nx // 2
    return np.repeat(blocks[:, bit:bit + 1], nx, axis=1).ravel()


def channel_labels(tools, mask):
    labels = [f'{freq}/{spacing} {component}' for freq, spacing in tools for component in UDAR_COMPONENTS]
    return [label for label, keep in zip(labels, np.asarray(mask, dtype=bool)) if keep]


def summarize(labels, observations, variance, phase, predictions, worst=6, comparisons=COMPARISONS):
    """Misfits against the data and pairwise prediction differences in noise units.

    ``predictions`` maps names such as '1d', '2d_invariant' and '2d_full' to
    retained-channel vectors already on the observed phase branch;
    ``comparisons`` names the (first, second) pairs to difference. Phase
    differences between two predictions are taken on the shorter arc.
    """
    variance = np.asarray(variance, dtype=float)
    components = np.array([label.split()[-1] for label in labels])
    report = {'data_misfit': {name: float(np.mean((np.asarray(value) - observations) ** 2 / variance))
                              for name, value in predictions.items()}}
    for key, (first, second) in comparisons.items():
        delta = align_phases(predictions[first], predictions[second], phase) - predictions[second]
        terms = delta ** 2 / variance / len(variance)
        total = float(terms.sum())
        order = np.argsort(terms)[::-1][:worst]
        report[key] = {
            'pair': [first, second], 'chi2_per_channel': total,
            'share_by_component': {name: float(terms[components == name].sum() / total) if total else 0.
                                   for name in dict.fromkeys(components)},
            'worst': [{'channel': labels[k], first: float(predictions[first][k]),
                       second: float(predictions[second][k]), 'observed': float(observations[k]),
                       'delta_sigma': float(delta[k] / np.sqrt(variance[k])),
                       'chi2_per_channel': float(terms[k])} for k in order]}
    return report


def parse_args(argv=None):
    defaults = SimBackendConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    defaults.add_cli(parser)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Directory written by sequential_matern_1d.py')
    parser.add_argument('--index', type=int, required=True, help='Logging point of the checkpoint')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--include-prior', action='store_true',
                        help="Also evaluate the member's prior field and its x-invariant copy")
    args = parser.parse_args(argv)
    return args, defaults.from_args(args)


def main(argv=None):
    args, config = parse_args(argv)
    run_dir, output = args.run_dir.resolve(), args.output_dir.resolve()
    study = json.loads((run_dir / 'study.json').read_text())
    saved = pickle.loads((run_dir / f'inversion_results_assim_{args.index}.pkl').read_bytes())
    manifest = json.loads((Path(study['reference']) / 'run.json').read_text())
    nx, ny, nz = saved['dims']
    dx, dy, dz = saved['spacing_ft']
    tools = [tuple(tool) for tool in saved['tools']]
    mask = np.asarray(saved['observation_mask'], dtype=bool)
    observations, variance = saved['observations'], saved['data_covariance']
    phase = phase_channels(mask, UDAR_COMPONENTS)
    member = best_member(saved['posterior_misfit'])
    field = np.asarray(saved['posterior_params'])[member]
    fields = {'2d_full': field, '2d_invariant': invariant_extension(field, nx, nz)}
    columns = {'1d': field}
    comparisons = dict(COMPARISONS)
    if args.include_prior:
        prior_field = np.asarray(saved['prior_params'])[:, member]
        fields.update({'2d_prior_full': prior_field,
                       '2d_prior_invariant': invariant_extension(prior_field, nx, nz)})
        columns['1d_prior'] = prior_field
        comparisons.update(PRIOR_COMPARISONS)
    # The 1D driver's raw-to-physical map.
    sequential.dims = (nx, ny, nz)
    sequential.dX, sequential.dY, sequential.dZ = dx, dy, dz
    sequential.tools = [str(tool) for tool in tools]
    sequential.max_ratio = float(manifest['prior_ratio_max'])
    sequential.log_rv_cap = float(np.log(saved['rv_hard_cap_ohmm']))
    sequential.anisotropy = saved['anisotropy']
    output.mkdir(parents=True, exist_ok=False)
    previous_cwd = Path.cwd()
    try:
        os.chdir(output)
        sequential.setup_simulators(study['benchmark'], sequential.tools)
        window = LocalWindow((nx, ny, nz), (dx, dy, dz))
        for index in range(args.index + 1):     # the driver's window path from the trajectory start
            shift, _ = window.update((sequential.WX[index], sequential.WY[index], sequential.TVD[index]))
        if any(not np.isclose(shift[key], saved['shift_ft'][key]) for key in ('x', 'y', 'z')):
            raise ValueError(f"Window shift {shift} differs from the checkpoint's {saved['shift_ft']}")
        sequential.current_shift = shift
        predictions = {
            name: align_phases(sequential.simulate_pred_only(
                value.reshape(2, nx, nz)[:, nx // 2].ravel(), args.index, '1D')[mask], observations, phase)
            for name, value in columns.items()}
        one_d = predictions['1d']
        sequential.remote_2d = make_remote2d(config, sequential.dims, dx, dy, dz, tools)
        sequential.remote_2d.jobs_dir = str(output / 'jobs')
        raw = np.asarray(sequential.remote_2d.run_pred_multi(
            [sequential._build_param_dict(value)[0] for value in fields.values()],
            args.index, sequential.current_shift, sequential.wellpath), dtype=float)
        if raw.shape != (len(fields), mask.size) or not np.isfinite(raw).all():
            raise RuntimeError(f'2D evaluation failed for at least one field: {raw.shape}, '
                               f'finite rows {np.isfinite(raw).all(axis=1).tolist()}')
        for name, row in zip(fields, raw):
            predictions[name] = align_phases(row[mask], observations, phase)
        labels = channel_labels(tools, mask)
        report = {'run_dir': str(run_dir), 'index': args.index, 'member': member,
                  'member_1d_misfit_checkpoint': float(saved['posterior_misfit'][member]),
                  'one_d_matches_checkpoint': bool(np.allclose(one_d, saved['posterior_predictions'][member])),
                  'likelihood_weight': study.get('likelihood_weight'),
                  'covariance_inflation': study.get('covariance_inflation'),
                  'simulator': asdict(config),
                  **summarize(labels, observations, variance, phase, predictions,
                              comparisons=comparisons)}
        np.savez(output / 'predictions.npz', labels=np.array(labels), observations=observations,
                 variance=variance, mask=mask, member=member,
                 **{f'field_{name}': value for name, value in fields.items()},
                 **{f'prediction_{name}': value for name, value in predictions.items()})
        (output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    finally:
        os.chdir(previous_cwd)
    misfit = report['data_misfit']
    print(f"Index {args.index}, member {member}: data chi2/n 1D {misfit['1d']:.4g}, "
          f"2D x-invariant {misfit['2d_invariant']:.4g}, 2D full {misfit['2d_full']:.4g}")
    if args.include_prior:
        print(f"  prior field: data chi2/n 1D {misfit['1d_prior']:.4g}, "
              f"2D x-invariant {misfit['2d_prior_invariant']:.4g}, 2D full {misfit['2d_prior_full']:.4g}")
    for key in comparisons:
        part = report[key]
        print(f"  {key} ({part['pair'][0]} vs {part['pair'][1]}): chi2/n {part['chi2_per_channel']:.4g}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
