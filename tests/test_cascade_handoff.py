import argparse
import json

import numpy as np
import pytest

import cascade_handoff
from rml_standalone import atomic_json, atomic_npz


def test_random_starts_come_from_best_quantile_and_top_is_best():
    misfits = np.arange(250.)[::-1]                 # member 249 is best
    starts = cascade_handoff.random_from_top(misfits, 10, .5, 4510)
    assert len(set(starts)) == 10 and np.all(misfits[starts] < np.median(misfits) + 1)
    np.testing.assert_array_equal(starts, cascade_handoff.random_from_top(misfits, 10, .5, 4510))
    best = cascade_handoff.top(misfits, 100)
    assert set(best) == set(range(150, 250))
    with pytest.raises(ValueError):
        cascade_handoff.random_from_top(misfits, 200, .5, 1)


def synthetic_cascade(tmp_path, members=20, cells=8, data=3):
    rng = np.random.default_rng(1)
    cascade = tmp_path / 'cascade'
    cascade.mkdir()
    fields = rng.normal(size=(2 * cells, members))
    observations, covariance = rng.normal(size=data), np.full(data, .5)
    predictions = observations[:, None] + rng.normal(size=(data, members)) * np.linspace(.1, 3, members)
    mask = np.ones(data, dtype=bool)
    atomic_npz(cascade / 'state-10-1d.npz', fields=fields, member_ids=np.arange(members))
    atomic_npz(cascade / 'mda-final-predictions.npz', predictions=predictions, observations=observations,
               covariance=covariance, mask=mask, fidelity='1D')
    atomic_json(cascade / 'study.json', {'mda_steps_executed': 10, 'mda_likelihood_weight': .5,
                                         'covariance_inflation': [20.] * 10, 'logging_index': 45})
    return cascade, fields, observations, covariance, mask


def synthetic_source(tmp_path, observations, covariance, mask):
    import pilot_rml_index45 as runner
    source = tmp_path / 'source'
    source.mkdir()
    atomic_npz(source / 'inputs.npz', observations=observations, variance=covariance, mask=mask,
               prior_centers=np.zeros((25, 16)), targets=np.zeros((25, 3)),
               well_X=np.zeros((50, 1)))
    study = {'index': 45, 'count': 25, 'dims': [2, 1, 4], 'spacing_ft': [5., 5., 5.],
             'prior': {'shape': [2, 4], 'spacing': [5., 5.], 'lengths': [[50., 50.], [80., 35.]],
                       'means': [1., 0.], 'stds': [1., .12]},
             'shift_ft': {'x': 0., 'y': 0., 'z': 0.}, 'tools': [['12kHz', '43ft']],
             'ratio_min': 1., 'ratio_max': 3., 'rh_limits': [-1., 3.69], 'log_rv_cap': 3.69,
             'optimizer_method': 'whitened_parallel_trust_region_lm_v1', 'optimizer': {},
             'resources': {'memory_gib': 48, 'walltime': '01:00:00'},
             'simulator': {'cpus': 8, 'ranks': 2, 'concurrency': 4, 'physical_cores': True},
             'inputs_sha256': runner.digest(source / 'inputs.npz')}
    atomic_json(source / 'study.json', study)
    return source


def test_rml_handoff_uses_selected_starts_inflated_noise_and_ensemble_prior(tmp_path):
    import pilot_rml_index45 as runner
    from lm_laplace import SpatialPrior
    cascade, fields, observations, covariance, mask = synthetic_cascade(tmp_path)
    source = synthetic_source(tmp_path, observations, covariance, mask)
    cascade_handoff.main(['--cascade', str(cascade), '--output', str(tmp_path / 'selection.json'),
                          '--rml-count', '4', '--enrml-count', '6'])
    selection = json.loads((tmp_path / 'selection.json').read_text())
    run = tmp_path / 'rml'
    runner.prepare_cascade(argparse.Namespace(run_dir=run, source=source, prior='cascade',
                                              selection=tmp_path / 'selection.json', noise_seed=3))
    study, inputs = runner.read_study(run)
    members = selection['rml']['members']
    np.testing.assert_array_equal(inputs['prior_centers'], fields[:, members].T)
    np.testing.assert_allclose(inputs['variance'], 2 * covariance)
    np.testing.assert_allclose(inputs['original_variance'], covariance)
    np.testing.assert_allclose(inputs['targets'] - inputs['observations'], inputs['measurement_noise'])
    assert study['count'] == 4 and study['data_inflation'] == pytest.approx(2.)
    prior = runner.load_prior(study, run)
    assert isinstance(prior, SpatialPrior)
    np.testing.assert_allclose(prior.mean, fields.mean(axis=1))
    script = (run / 'rml-array.sbatch').read_text()
    assert '#SBATCH --array=0-3\n' in script and '#SBATCH --cpus-per-task=16\n' in script


def test_enrml_handoff_takes_best_members_and_inflates_data_variance(tmp_path):
    import enrml_index45
    cascade, fields, observations, covariance, mask = synthetic_cascade(tmp_path)
    source = synthetic_source(tmp_path, observations, covariance, mask)
    cascade_handoff.main(['--cascade', str(cascade), '--output', str(tmp_path / 'selection.json'),
                          '--rml-count', '4', '--enrml-count', '6'])
    run = tmp_path / 'enrml'
    enrml_index45.prepare(argparse.Namespace(
        source=source, run_dir=run, count=250, seed=1, data_seed=2, max_iter=30, lambda_factor=10.,
        data_misfit_tol=.01, energy=.95, concurrency=6, nodes=1, memory_gib=64, walltime='03:00:00',
        selection=tmp_path / 'selection.json'))
    config = json.loads((run / 'config.json').read_text())
    best = json.loads((tmp_path / 'selection.json').read_text())['enrml']['members']
    with np.load(run / 'prior.npz') as prior:
        np.testing.assert_array_equal(prior['lnrh'], fields[:8, best])
        np.testing.assert_array_equal(prior['latent'], fields[8:, best])
    assert config['count'] == 6 and config['data_inflation'] == pytest.approx(2.)
    keys_da, keys_en = enrml_index45.pet_keys(config, {'observations': observations, 'variance': covariance},
                                              run / 'prior.npz', run / 'out')
    np.testing.assert_allclose(keys_da['datavar'][1], 2 * covariance)
    assert keys_en['ne'] == 6


def test_rml_original_prior_mode_reuses_source_problems_and_only_sets_starts(tmp_path):
    import pilot_rml_index45 as runner
    from background_prior import BackgroundPrior
    cascade, fields, observations, covariance, mask = synthetic_cascade(tmp_path)
    source = synthetic_source(tmp_path, observations, covariance, mask)
    with np.load(source / 'inputs.npz') as saved:
        source_inputs = {key: saved[key].copy() for key in saved.files}
    rng = np.random.default_rng(9)
    source_inputs.update(prior_centers=rng.normal(size=(25, 16)), measurement_noise=rng.normal(size=(25, 3)))
    source_inputs['targets'] = observations + source_inputs['measurement_noise']
    atomic_npz(source / 'inputs.npz', **source_inputs)
    study = json.loads((source / 'study.json').read_text())
    study['inputs_sha256'] = runner.digest(source / 'inputs.npz')
    atomic_json(source / 'study.json', study)
    cascade_handoff.main(['--cascade', str(cascade), '--output', str(tmp_path / 'selection.json'),
                          '--rml-count', '4', '--enrml-count', '6'])
    members = json.loads((tmp_path / 'selection.json').read_text())['rml']['members']
    run = tmp_path / 'rml-original'
    runner.prepare_cascade(argparse.Namespace(run_dir=run, source=source, prior='original',
                                              selection=tmp_path / 'selection.json', noise_seed=3))
    prepared, inputs = runner.read_study(run)
    for key in ('prior_centers', 'measurement_noise', 'targets'):
        np.testing.assert_array_equal(inputs[key], source_inputs[key][:4])
    np.testing.assert_array_equal(inputs['start_models'], fields[:, members].T)
    np.testing.assert_array_equal(inputs['variance'], covariance)
    assert prepared['data_inflation'] == 1. and prepared['count'] == 4
    assert isinstance(runner.load_prior(prepared, run), BackgroundPrior)
    assert not (run / 'prior-ensemble.npz').exists()
