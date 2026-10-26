"""Plot the forward-verified 2D-only index-31 posterior against Benchmark-3."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_curtain import posterior_moments
from transition_mda_study import checkpoint, overlap_weights, reference_truth, rh_rmse


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / 'runs/2d-only-index31-independent-rh40-rv40-20261017'


def load_posterior(run_dir):
    study = json.loads((run_dir / 'study.json').read_text())
    result = json.loads((run_dir / 'result.json').read_text())
    if study['logging_index'] != 31 or result['status'] != 'complete':
        raise ValueError('Require a completed 2D-only index-31 inversion')
    final_step = result['mda_updates_completed']
    with np.load(run_dir / f'state-{final_step:02d}.npz') as state, np.load(
            run_dir / f'evaluation-{final_step:02d}.npz') as evaluation:
        members = state['member_ids']
        valid = evaluation['valid']
        if (not np.array_equal(members, evaluation['requested_ids'])
                or valid.shape != members.shape
                or not np.array_equal(members[valid], result['history'][-1]['surviving_members'])
                or int(valid.sum()) != result['final_members']):
            raise ValueError('Final 2D evaluation and posterior member identities disagree')
        if not np.isfinite(evaluation['predictions'][:, valid]).all():
            raise ValueError('Selected posterior has non-finite 2D forward predictions')
        parameters = state['fields'][:, valid].copy()
    return study, result, members[valid].tolist(), parameters


def benchmark_curtain(benchmark, study):
    x, z, truth_rh, wellpath = reference_truth(benchmark)
    nx, _, nz = study['dims']
    dx, _, dz = study['spacing_ft']
    shift = study['shift_ft']
    x_edges = shift['x'] + np.arange(nx + 1) * dx
    z_edges = shift['z'] + np.arange(nz + 1) * dz
    wx = overlap_weights(x, x_edges, float(np.median(np.diff(x))))
    wz = overlap_weights(z, z_edges, float(np.median(np.diff(z))))
    with h5py.File(benchmark / 'globalmodel.h5', 'r') as handle:
        mid = handle['formation/Rv'].shape[1] // 2
        truth_rv = handle['formation/Rv'][:, mid, :]
    truth = {'Rh': wx @ truth_rh.T @ wz.T, 'Rv': wx @ truth_rv.T @ wz.T}
    well_x, well_z = (wellpath[key][study['logging_index'], 0] * 3.28084
                      for key in ('X', 'Z'))
    return truth, x_edges, z_edges, (well_x, well_z), (x, z, truth_rh, wellpath)


def render(run_dir, output, dpi):
    study, result, member_ids, parameters = load_posterior(run_dir)
    reference = Path(study['reference'])
    baseline = json.loads((reference / 'run.json').read_text())
    benchmark = Path(baseline['benchmark'])
    saved = checkpoint(reference, 31)
    truth, x_edges, z_edges, well, rh_truth = benchmark_curtain(benchmark, study)
    cap = study.get('rv_cap_ohmm_effective', study['prior_parameters']['prior_rv_cap_ohmm'])
    bounds = baseline['rv_parameterization']['ratio_bounds']
    mean_rh, mean_rv, variance_rh, variance_rv, _ = posterior_moments(
        parameters.T, study['dims'], bounds, cap)
    rmse_rh = rh_rmse(parameters, saved, rh_truth)
    if not np.isclose(rmse_rh, result['final_rh_rmse_log10'], atol=1e-10):
        raise ValueError('Curtain Rh RMSE disagrees with the verified result')
    moments = {'Rh': (mean_rh / np.log(10.), np.sqrt(variance_rh)),
               'Rv': (mean_rv / np.log(10.), np.sqrt(variance_rv))}
    ref_logs = {field: np.log10(truth[field]) for field in ('Rh', 'Rv')}
    values = np.concatenate([ref_logs['Rh'].ravel(), ref_logs['Rv'].ravel(),
                             moments['Rh'][0].ravel(), moments['Rv'][0].ravel()])
    vmin, vmax = np.percentile(values, [1, 99])
    std_max = max(float(np.percentile(moments[field][1], 99)) for field in ('Rh', 'Rv'))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True,
                             sharex=True, sharey=True)
    image = spread = None
    details = {}
    for row, field in enumerate(('Rh', 'Rv')):
        model, std = moments[field]
        details[field] = {'benchmark_rmse_log10': float(np.sqrt(np.mean(
            (model - ref_logs[field]) ** 2))),
                          'posterior_std_log10_p95': float(np.percentile(std, 95))}
        for col, (array, title) in enumerate(((ref_logs[field], 'Benchmark'),
                                               (model, 'Posterior mean'),
                                               (std, 'Posterior std'))):
            ax = axes[row, col]
            artist = ax.pcolormesh(x_edges, z_edges, array.T, shading='flat',
                                   cmap='magma' if col == 2 else 'viridis',
                                   vmin=0 if col == 2 else vmin,
                                   vmax=std_max if col == 2 else vmax)
            if col == 2:
                spread = artist
            else:
                image = artist
            ax.plot(well[0], well[1], 'x', color='white', ms=7, mew=1.7)
            ax.set_title(f'{title} — {field}')
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(z_edges[-1], z_edges[0])
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X [ft]')
            if col == 0:
                ax.set_ylabel('TVD [ft]')
    fig.colorbar(image, ax=axes[:, :2].ravel().tolist(), shrink=.72,
                 label='log10 resistivity [ohm m]')
    fig.colorbar(spread, ax=axes[:, 2].ravel().tolist(), shrink=.72,
                 label='Posterior std of log10 resistivity')
    fig.suptitle(f"Index 31 • 2D-only posterior • {len(member_ids)}/25 forward-valid members"
                 f" • reduced misfit {result['final_reduced_misfit']:.2f}", fontsize=14)
    fig.savefig(output, dpi=dpi)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)
    metadata = {'index': 31, 'source': str(run_dir), 'posterior_member_ids': member_ids,
                'members': len(member_ids), 'rv_cap_ohmm': cap,
                'reduced_misfit': result['final_reduced_misfit'],
                'rmse_log10_rh': rmse_rh, 'details': details,
                'png': str(output), 'svg': str(output.with_suffix('.svg'))}
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dpi', type=int, default=190)
    args = parser.parse_args(argv)
    if args.dpi < 40:
        parser.error('Require dpi >= 40')
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else run_dir / 'posterior_curtain.png'
    if output.suffix.lower() != '.png' or not output.parent.exists():
        parser.error('Output must be a PNG path in an existing directory')
    metadata = render(run_dir, output, args.dpi)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
