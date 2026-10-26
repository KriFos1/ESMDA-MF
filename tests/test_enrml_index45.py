import json
import subprocess

import numpy as np
import pytest

pytest.importorskip('pipt')

from enrml_index45 import PetForward, assimilate, write_sbatch
from rml_standalone import atomic_npz


def problem(tmp_path, count=40, failing=()):
    rng = np.random.default_rng(3)
    cells = 6
    matrix = rng.normal(size=(5, 2 * cells)) / 2
    truth = rng.normal(size=2 * cells)
    variance = np.full(5, .01)
    observations = np.tanh(matrix @ truth) + rng.normal(size=5) * .1
    prior = rng.normal(size=(2 * cells, count))
    atomic_npz(tmp_path / 'prior.npz', lnrh=prior[:cells], latent=prior[cells:])
    config = {'count': count, 'index': 45, 'dims': [3, 1, 2], 'data_seed': 7,
              'prior': {'means': [0., 0.], 'stds': [1., 1.]},
              'pet': {'max_iter': 8, 'lambda': 'auto', 'lambda_factor': 10.,
                      'data_misfit_tol': .01, 'energy': .95},
              'resources': {'concurrency': 4}}
    calls = []

    def forward(raws):
        calls.append(raws.shape[1])
        predictions = np.tanh(matrix @ raws)
        if len(calls) == 1:
            predictions[:, list(failing)] = np.nan
        return predictions
    inputs = {'observations': observations, 'variance': variance}
    return config, inputs, forward, calls


def test_pet_lmenrml_runs_with_auto_lambda_and_single_factor(tmp_path):
    config, inputs, forward, calls = problem(tmp_path)
    output = tmp_path / 'run'
    output.mkdir()
    summary, simulator = assimilate(config, inputs, tmp_path / 'prior.npz', forward, output)
    history = json.loads((output / 'history.json').read_text())
    assert set(calls) == {40} and len(calls) == summary['forecast_batches']
    assert summary['auto_lambda'] == pytest.approx(.5 * summary['prior_data_misfit'] / 5)
    assert summary['final_data_misfit'] < summary['prior_data_misfit']
    for row in history:
        if row['converged']:
            continue
        if row['success'] and row['data_misfit_std'] < row['prior_data_misfit']:
            assert row['lambda_next'] in (pytest.approx(row['lambda_used'] / 10.),
                                          pytest.approx(row['lambda_used']))
        if not row['success']:
            assert row['lambda_next'] == pytest.approx(row['lambda_used'] * 10.)
    with np.load(output / 'enrml-result.npz') as result:
        assert result['raw_models'].shape == (12, 40)
        assert np.isfinite(result['predictions']).all()
        assert result['perturbed_observations'].shape == (5, 40)
    assert any(row['success'] for row in history)
    assert summary['unmatched_final_predictions'] == 0


def test_final_predictions_belong_to_the_accepted_states(tmp_path):
    config, inputs, forward, _ = problem(tmp_path)
    output = tmp_path / 'run'
    output.mkdir()
    assimilate(config, inputs, tmp_path / 'prior.npz', forward, output)
    with np.load(output / 'enrml-result.npz') as result:
        recomputed = forward(result['raw_models'])
        np.testing.assert_allclose(result['predictions'], recomputed, atol=1e-12)


def test_failed_prior_solves_are_replaced_by_pet(tmp_path):
    config, inputs, forward, _ = problem(tmp_path, failing=(0, 3))
    output = tmp_path / 'run'
    output.mkdir()
    summary, simulator = assimilate(config, inputs, tmp_path / 'prior.npz', forward, output)
    assert summary['failed_solves'] == 2
    assert simulator.batches[0][2][[0, 3]].all()
    with np.load(output / 'forecasts/forecast-000.npz') as first:
        assert first['failed'].sum() == 2


def test_pet_forward_reports_failures_as_false(tmp_path):
    def forward(raws):
        predictions = raws[:2].copy()
        predictions[:, 1] = np.nan
        return predictions
    simulator = PetForward(forward, tmp_path / 'archive', 2)
    states = [{'lnrh': np.array([1., 2.]), 'latent': np.array([3., 4.])},
              {'lnrh': np.array([5., 6.]), 'latent': np.array([7., 8.])}]
    result = simulator.run_fwd_batch(states, [0, 1])
    assert result[1] is False
    np.testing.assert_array_equal(result[0][0]['udar'], [1., 2.])
    found = simulator.predictions_for(np.array([[1., 5.], [2., 6.], [3., 7.], [4., 8.]]))
    np.testing.assert_array_equal(found[:, 0], [1., 2.])
    assert np.isnan(found[:, 1]).all()


def test_sbatch_requests_physical_core_steps_per_node(tmp_path):
    config = {'index': 45, 'simulator': {'ranks': 2, 'cpus': 8},
              'resources': {'concurrency': 16, 'nodes': 4, 'memory_gib_per_node': 64,
                            'walltime': '02:00:00'}}
    write_sbatch(tmp_path, config)
    script = (tmp_path / 'enrml.sbatch').read_text()
    for line in ('#SBATCH --nodes=4\n', '#SBATCH --ntasks-per-node=8\n', '#SBATCH --cpus-per-task=16\n',
                 '#SBATCH --mem=64G\n'):
        assert line in script
    assert subprocess.run(['bash', '-n', str(tmp_path / 'enrml.sbatch')]).returncode == 0
