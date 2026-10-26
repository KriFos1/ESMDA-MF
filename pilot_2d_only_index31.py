"""Ten-step, 2D-only ES-MDA at Benchmark-3 logging index 31.

Draw 25 unconditioned spatial prior members with independent ln(Rh) and
anisotropy-latent fields (optional pointwise cross-correlation). Every assimilated
prediction is computed by the 2D simulator; no 0D or 1D forward is used.
Stage files make a failed/interrupted HPC batch resumable without reusing a
different draw or silently replacing members whose 2D evaluation failed.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from esmda_core import (LOG_RH_LIMITS, LocalWindow, clamp_log_rh, measurement_misfit,
                        noise_floor_mask, observation_vectors)
from esmda_remote2d import SimBackendConfig, make_remote2d
import run_DA_sequential as sequential
from transition_mda_study import (DEFAULT_REFERENCE, checkpoint, inflation_schedule,
                                  reference_truth, rh_rmse, update)


ROOT = Path(__file__).resolve().parent
INDEX = 31
MEMBERS = 25
LOCALIZATION_THRESHOLD = .3
LATENT_RH_CORRELATION = 0.
RH_CAP_OHMM = 40.
RV_CAP_OHMM = 40.
RH_SAMPLER = 'fast_gaussian'
RH_CORRELATION_X_FT = RH_CORRELATION_Z_FT = 50.


def rh_prior_moments(manifest):
    ln10 = np.log(10.)
    background = manifest['prior_background'] * ln10
    reservoir = manifest['prior_reservoir'] * ln10
    mode_std = manifest['prior_mode_std'] * ln10
    fraction = manifest['prior_reservoir_fraction']
    mean = fraction * reservoir + (1. - fraction) * background
    variance = mode_std ** 2 + fraction * (1. - fraction) * (reservoir - background) ** 2
    return mean, np.sqrt(variance)


def draw_prior(manifest, seed, correlation=LATENT_RH_CORRELATION,
               rh_sampler=RH_SAMPLER):
    """Draw Gaussian fields, using geostat for Rh and the original latent prior."""
    if manifest['prior_style'] != 'gaussian' or manifest['prior_projection'] != 'off':
        raise ValueError('This pilot requires the reference Gaussian prior without projection')
    if not np.isfinite(correlation) or not 0 <= correlation < 1:
        raise ValueError('Require 0 <= latent/Rh prior correlation < 1')
    nx, ny, nz = manifest['dims']
    sequential.dims = nx, ny, nz
    mean_rh, std_rh = rh_prior_moments(manifest)
    if rh_sampler == 'fast_gaussian':
        from geostat.gaussian_sim import fast_gaussian

        np.random.seed(seed)
        rh = mean_rh + fast_gaussian(
            np.array([nz, nx]), np.array([std_rh]),
            np.array([RH_CORRELATION_Z_FT / manifest['dz_ft'],
                      RH_CORRELATION_X_FT / manifest['dx_ft']]), num_samples=MEMBERS)
        np.random.seed(seed + 1)
        latent = manifest['rv_parameterization']['latent_std'] * sequential._ar1_field(
            nx, nz, MEMBERS,
            np.exp(-manifest['dx_ft'] / manifest['prior_latent_correlation_x_ft']),
            np.exp(-manifest['dz_ft'] / manifest['prior_latent_correlation_z_ft']))
        samples = np.vstack([rh, latent.reshape(nx * nz, MEMBERS)])
    elif rh_sampler == 'ar1':
        np.random.seed(seed)
        ln10 = np.log(10.)
        samples, _, _ = sequential.sample_prior_2d(
            MEMBERS, None,
            manifest['prior_background'] * ln10,
            manifest['prior_reservoir'] * ln10,
            manifest['prior_mode_std'] * ln10,
            manifest['prior_reservoir_fraction'],
            manifest['prior_switch_prob'], projection=False, prior_style='gaussian',
            dx=manifest['dx_ft'], dz=manifest['dz_ft'],
            corr_x_ft=manifest['prior_correlation_x_ft'],
            corr_z_ft=manifest['prior_correlation_z_ft'],
            latent_corr_z_ft=manifest['prior_latent_correlation_z_ft'],
            latent_corr_x_ft=manifest['prior_latent_correlation_x_ft'])
    else:
        raise ValueError(f'Unknown Rh sampler {rh_sampler}')
    if correlation:
        cells = nx * nz
        latent_std = float(manifest['rv_parameterization']['latent_std'])
        if not np.isclose(latent_std, sequential.ratio_latent_std):
            raise ValueError('Reference latent prior standard deviation changed')
        samples[cells:] = (correlation * latent_std * (samples[:cells] - mean_rh) / std_rh
                           + np.sqrt(1. - correlation ** 2) * samples[cells:])
    samples = clamp_log_rh(samples)
    if samples.shape != (2 * nx * nz, MEMBERS) or not np.isfinite(samples).all():
        raise ValueError('Prior sampler did not yield 25 finite fields')
    return samples


def cap_rh(state, maximum_ohmm):
    """Apply the physical Rh ceiling to the parameter state before a 2D call."""
    rh = state[:state.shape[0] // 2]
    np.minimum(rh, np.log(maximum_ohmm), out=rh)
    return state


def write_npz(path, **values):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)


def read_npz(path):
    with np.load(path) as saved:
        return {name: saved[name] for name in saved.files}


def summarize_step(step, requested, surviving, predictions, observations,
                   covariance, fields, reference, truth, alpha):
    individual = [measurement_misfit(predictions[:, col], observations, covariance)
                  for col in range(predictions.shape[1])]
    return {'evaluation': step, 'mda_updates_completed': step,
            'alpha_for_next_update': float(alpha) if alpha is not None else None,
            'attempted_members': requested.tolist(),
            'failed_members': sorted(int(member) for member in
                                     set(requested) - set(surviving)),
            'surviving_members': surviving.tolist(),
            'member_count': int(len(surviving)),
            'ensemble_mean_reduced_misfit': float(measurement_misfit(
                predictions.mean(axis=1), observations, covariance)),
            'best_member_reduced_misfit': float(min(individual)),
            'median_member_reduced_misfit': float(np.median(individual)),
            'rh_rmse_log10': rh_rmse(fields, reference, truth)}


def main(argv=None):
    defaults = SimBackendConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    defaults.add_cli(parser)
    parser.set_defaults(concurrency=MEMBERS)
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument('--rh-sampler', choices=['fast_gaussian', 'ar1'], default=RH_SAMPLER,
                        help='fast_gaussian uses 50/50-ft Gaussian variogram; ar1 replays old prior')
    parser.add_argument('--latent-rh-correlation', type=float, default=LATENT_RH_CORRELATION,
                        help='Pointwise prior Corr(latent, ln Rh); default 0 (independent)')
    parser.add_argument('--rh-cap-ohmm', type=float, default=RH_CAP_OHMM,
                        help='Maximum Rh for sampled and updated 2D parameter states; default 40 Ohm-m')
    parser.add_argument('--rv-cap-ohmm', type=float, default=RV_CAP_OHMM,
                        help='Maximum Rv used by the 2D forward; default 40 Ohm-m')
    parser.add_argument('--sample-only', action='store_true',
                        help='Save the fresh prior and check alignment without submitting HPC jobs')
    args = parser.parse_args(argv)
    if not 1 <= args.concurrency <= MEMBERS:
        parser.error('Require 1 <= concurrency <= 25')
    if not np.isfinite(args.latent_rh_correlation) or not 0 <= args.latent_rh_correlation < 1:
        parser.error('Require 0 <= latent-rh-correlation < 1')
    if not np.isfinite(args.rh_cap_ohmm) or args.rh_cap_ohmm < np.exp(LOG_RH_LIMITS[0]):
        parser.error('Require a finite Rh cap above the existing lower Rh bound')
    if not np.isfinite(args.rv_cap_ohmm) or args.rv_cap_ohmm <= 0:
        parser.error('Require a positive finite Rv cap')
    if not args.sample_only and args.sim_backend != 'slurm':
        parser.error('The 2D-only inversion requires --sim-backend slurm')

    reference_dir = args.reference.resolve()
    manifest = json.loads((reference_dir / 'run.json').read_text())
    reference_path = reference_dir / f'inversion_results_assim_{INDEX}.pkl'
    reference = checkpoint(reference_dir, INDEX)
    reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    benchmark = Path(manifest['benchmark'])
    truth = reference_truth(benchmark)
    nx, ny, nz = manifest['dims']
    dx, dy, dz = manifest['spacing_ft']
    sequential.dims = nx, ny, nz
    sequential.dX, sequential.dY, sequential.dZ = dx, dy, dz
    sequential.max_ratio = float(manifest['prior_ratio_max'])
    sequential.log_rv_cap = float(np.log(args.rv_cap_ohmm))
    sequential.wellpath = truth[3]
    window = LocalWindow((nx, ny, nz), (dx, dy, dz))
    shift = None
    for point in range(INDEX + 1):
        position = tuple(truth[3][key][point, 0] * sequential.meter_to_feet
                         for key in ('X', 'Y', 'Z'))
        shift, _ = window.update(position)
    if not all(np.isclose(shift[key], reference['shift_ft'][key], rtol=0, atol=1e-6)
               for key in ('x', 'y', 'z')):
        raise ValueError('Index-31 moving-window geometry disagrees with reference')
    sequential.current_shift = shift

    data = pd.read_pickle(ROOT / manifest['data'])
    variances = pd.read_pickle(ROOT / manifest['variance'])
    tools = list(data.columns)
    observations, covariance = observation_vectors(data, variances, INDEX, tools)
    shallow = np.zeros(len(observations), dtype=bool)
    shallow[sequential.component_positions(len(observations))] = True
    mask = ~(noise_floor_mask(observations, covariance) | shallow)
    if not np.array_equal(mask, reference['observation_mask']) or not np.array_equal(
            observations[mask], reference['observations']):
        raise ValueError('Index-31 data or independently reconstructed channel mask changed')
    observations, covariance = observations[mask], covariance[mask]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = defaults.from_args(args)
    study = {'logging_index': INDEX, 'reference_checkpoint_sha256': reference_hash,
             'reference': str(reference_dir), 'seed': args.seed,
             'prior': (('25 fresh unconditioned draws of saved Gaussian spatial prior'
                        if args.latent_rh_correlation == 0 else
                        '25 fresh Gaussian spatial prior draws with latent/Rh cross-correlation')
                       if args.rh_sampler == 'ar1' else
                       '25 fresh geostat.fast_gaussian Rh fields and independent AR1 latent fields'),
             'prior_parameters': {key: manifest[key] for key in (
                 'prior_background', 'prior_reservoir', 'prior_mode_std',
                 'prior_reservoir_fraction', 'prior_correlation_x_ft',
                 'prior_correlation_z_ft', 'prior_latent_correlation_x_ft',
                 'prior_latent_correlation_z_ft', 'prior_ratio_max', 'prior_rv_cap_ohmm')},
             'dims': [nx, ny, nz], 'spacing_ft': [dx, dy, dz],
             'shift_ft': {name: float(shift[name]) for name in ('x', 'y', 'z')},
             'observation_count': int(len(observations)),
             'covariance_inflation': inflation_schedule().tolist(),
              'localization': {'type': 'sigm', 'correlation_threshold': LOCALIZATION_THRESHOLD},
              'simulator': asdict(config)}
    if args.latent_rh_correlation:
        study['latent_rh_correlation'] = args.latent_rh_correlation
    if args.rh_sampler == 'fast_gaussian':
        mean_rh, std_rh = rh_prior_moments(manifest)
        study['rh_sampler'] = {'name': 'geostat.gaussian_sim.fast_gaussian',
                               'mean_ln': float(mean_rh), 'variance_ln': float(std_rh ** 2),
                               'correlation_ft': {'x': RH_CORRELATION_X_FT,
                                                  'z': RH_CORRELATION_Z_FT},
                               'correlation_cells': {'x': RH_CORRELATION_X_FT / dx,
                                                     'z': RH_CORRELATION_Z_FT / dz},
                               'variogram': 'gaussian',
                               'dimension_order': ['z', 'x'], 'latent_seed': args.seed + 1}
    if args.rh_cap_ohmm < np.exp(LOG_RH_LIMITS[1]):
        study['rh_cap_ohmm_effective'] = float(args.rh_cap_ohmm)
    if not np.isclose(args.rv_cap_ohmm, manifest['prior_rv_cap_ohmm']):
        study['reference_rv_cap_ohmm'] = float(manifest['prior_rv_cap_ohmm'])
        study['prior_parameters']['prior_rv_cap_ohmm'] = float(args.rv_cap_ohmm)
        study['rv_cap_ohmm_effective'] = float(args.rv_cap_ohmm)
    study_path = output / 'study.json'
    serialized_study = json.dumps(study, indent=2)
    if study_path.exists():
        if json.loads(study_path.read_text()) != json.loads(serialized_study):
            raise ValueError('Existing study configuration differs; use a fresh output directory')
    else:
        study_path.write_text(serialized_study)
    initial_path = output / 'state-00.npz'
    if not initial_path.exists():
        write_npz(initial_path, fields=cap_rh(draw_prior(manifest, args.seed,
                                                        args.latent_rh_correlation,
                                                        args.rh_sampler),
                                              args.rh_cap_ohmm),
                  member_ids=np.arange(MEMBERS))
    if args.sample_only:
        fields = read_npz(initial_path)['fields']
        cells = nx * nz
        print(json.dumps({'prior_members': fields.shape[1],
                          'prior_rh_rmse_log10': rh_rmse(fields, reference, truth),
                          'sample_latent_rh_correlation': float(np.corrcoef(
                              fields[:cells].ravel(), fields[cells:].ravel())[0, 1]),
                          'output_dir': str(output)}), flush=True)
        return

    import ast

    adapter = make_remote2d(config, (nx, ny, nz), dx, dy, dz,
                            [ast.literal_eval(str(tool)) for tool in tools])
    adapter.jobs_dir = str(output / 'jobs')
    history = []
    inflation = inflation_schedule()
    for step in range(len(inflation) + 1):
        current = read_npz(output / f'state-{step:02d}.npz')
        fields, member_ids = current['fields'], current['member_ids']
        evaluation_path = output / f'evaluation-{step:02d}.npz'
        if evaluation_path.exists():
            evaluated = read_npz(evaluation_path)
            if not np.array_equal(member_ids, evaluated['requested_ids']):
                raise ValueError(f'Step {step}: saved member identities changed')
            predicted, valid = evaluated['predictions'], evaluated['valid']
        else:
            models = [sequential._build_param_dict(fields[:, member])[0]
                      for member in range(fields.shape[1])]
            responses = adapter.run_pred_multi(models, INDEX, shift, truth[3])
            predicted = np.asarray(responses, dtype=float)[:, mask].T
            if predicted.shape != (len(observations), len(member_ids)):
                raise ValueError(f'Step {step}: 2D batch response has wrong shape')
            valid = np.isfinite(predicted).all(axis=0)
            write_npz(evaluation_path, requested_ids=member_ids,
                      predictions=predicted, valid=valid)
        kept, surviving = fields[:, valid], member_ids[valid]
        if len(surviving) < 2:
            last = history[-1] if history else {}
            result = {'status': 'insufficient_2d_members', 'failed_at_evaluation': step,
                      'mda_updates_completed': step, 'initial_members': MEMBERS,
                      'requested_members': member_ids.tolist(),
                      'surviving_members': surviving.tolist(),
                      'last_forward_verified_updates': last.get('mda_updates_completed'),
                      'last_forward_verified_reduced_misfit': last.get(
                          'ensemble_mean_reduced_misfit'),
                      'last_forward_verified_rh_rmse_log10': last.get('rh_rmse_log10'),
                      'initial_reduced_misfit': (history[0]['ensemble_mean_reduced_misfit']
                                                 if history else None),
                      'total_2d_forward_attempts': (sum(len(entry['attempted_members'])
                                                        for entry in history) + len(member_ids)),
                      'unspent_likelihood_weight': float(np.reciprocal(inflation[step:]).sum()),
                      'history': history}
            (output / 'result.json').write_text(json.dumps(result, indent=2))
            print(f'Step {step}: only {len(surviving)} members survived; stopping', flush=True)
            return
        predicted = predicted[:, valid]
        entry = summarize_step(step, member_ids, surviving, predicted, observations,
                               covariance, kept, reference, truth,
                               inflation[step] if step < len(inflation) else None)
        history.append(entry)
        (output / f'step-{step:02d}.json').write_text(json.dumps(entry, indent=2))
        print(f"Index 31, 2D evaluation {step}: {len(surviving)}/{MEMBERS} valid, "
              f"mean misfit={entry['ensemble_mean_reduced_misfit']:.4g}, "
              f"Rh RMSE={entry['rh_rmse_log10']:.4g}", flush=True)
        if step < len(inflation):
            next_path = output / f'state-{step + 1:02d}.npz'
            if not next_path.exists():
                rng = np.random.default_rng(args.seed + 1000 + step)
                updated = update(kept, predicted, observations, covariance,
                                 inflation[step], rng, nx * nz, LOCALIZATION_THRESHOLD)
                cap_rh(updated, args.rh_cap_ohmm)
                if not np.isfinite(updated).all():
                    raise ValueError(f'Step {step}: ES-MDA update contains non-finite parameters')
                write_npz(next_path, fields=updated, member_ids=surviving)
    result = {'status': 'complete', 'mda_updates_completed': len(inflation),
              'initial_members': MEMBERS, 'final_members': len(surviving),
              'total_2d_forward_attempts': sum(len(entry['attempted_members'])
                                               for entry in history),
              'initial_reduced_misfit': history[0]['ensemble_mean_reduced_misfit'],
              'final_reduced_misfit': history[-1]['ensemble_mean_reduced_misfit'],
              'initial_rh_rmse_log10': history[0]['rh_rmse_log10'],
              'final_rh_rmse_log10': history[-1]['rh_rmse_log10'],
              'history': history}
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(f"Completed ten 2D ES-MDA updates: misfit {result['initial_reduced_misfit']:.4g} "
          f"-> {result['final_reduced_misfit']:.4g}", flush=True)


if __name__ == '__main__':
    main()
