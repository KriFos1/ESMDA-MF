import numpy as np
import pytest

from esmda_core import update_members_once
import tv_regularization as tv


def test_tv_constant_and_grid_spacing_contract():
    value, gradient = tv.value_and_gradient(np.full(5, 1.7), (5,), (2.,))
    assert value == pytest.approx(0.)
    np.testing.assert_allclose(gradient, 0.)
    column = np.array([0., 0., 2., 2.])
    one_d, _ = tv.value_and_gradient(column, (4,), (5.,))
    field = np.tile(column, (3, 1))
    two_d, _ = tv.value_and_gradient(field, (3, 4), (7., 5.))
    assert two_d == pytest.approx(one_d)
    exact_coarse, _ = tv.value_and_gradient(column, (4,), (5.,), epsilon=1e-9)
    refined, _ = tv.value_and_gradient(np.repeat(column, 2), (8,), (2.5,), epsilon=1e-9)
    assert refined == pytest.approx(exact_coarse, rel=1e-7)


def test_tv_gradient_matches_finite_differences():
    rng = np.random.default_rng(9)
    values = rng.normal(size=12)
    direction = rng.normal(size=12)
    _, gradient = tv.value_and_gradient(values, (3, 4), (2., 5.), epsilon=.03)
    step = 1e-6
    plus, _ = tv.value_and_gradient(values + step * direction, (3, 4), (2., 5.), .03)
    minus, _ = tv.value_and_gradient(values - step * direction, (3, 4), (2., 5.), .03)
    assert gradient @ direction == pytest.approx((plus - minus) / (2 * step), rel=1e-6)


def test_prox_removes_small_oscillations_without_snapping_to_levels():
    values = np.array([.1, -.1, .1, -.1, 2.1, 1.9, 2.1, 1.9])
    strength = .12
    result = tv.prox(values, strength, (1.,), iterations=20)
    assert result.mean() == pytest.approx(values.mean(), abs=1e-8)
    assert np.ptp(result[:4]) < np.ptp(values[:4])
    assert result[4] - result[3] > 1.
    before = strength * tv.value_and_gradient(values, values.shape, (1.,))[0]
    after = .5 * np.sum((result - values) ** 2) + strength * tv.value_and_gradient(
        result, values.shape, (1.,))[0]
    assert after < before
    np.testing.assert_array_equal(tv.prox(values, 0., (1.,)), values)


def test_member_prox_keeps_anisotropy_and_individual_structure():
    ensemble = np.array([[0., 2.], [.2, 1.8], [2., 0.], [1.8, .2],
                        [.1, .3], [.2, .4], [.3, .5], [.4, .6]])
    result = tv.regularize_columns(ensemble, .1, 5.)
    np.testing.assert_array_equal(result[4:], ensemble[4:])
    assert not np.allclose(result[:4, 0], result[:4, 1])


def test_fixed_jacobian_tv_reduces_roughness_and_preserves_fixed_cells():
    params = np.zeros(8)
    jacobian = np.hstack([np.eye(4), np.zeros((4, 4))])
    datum = np.array([0., 2.1, 1.9, 0.])
    active = np.ones(8, dtype=bool)
    active[[0, 4]] = False
    kwargs = dict(shape=(4,), spacing=(1.,), active_mask=active, iterations=12,
                  amplitude_std=10., max_step=5., bounds=(-1., 4.))
    zero = tv.linearized_step(params, np.zeros(4), params, jacobian, datum, np.ones(4),
                              strength=0., **kwargs)
    regularized = tv.linearized_step(params, np.zeros(4), params, jacobian, datum, np.ones(4),
                                     strength=.3, **kwargs)
    assert regularized[0] == regularized[4] == 0.
    assert tv.value_and_gradient(regularized[:4], (4,), (1.,))[0] < tv.value_and_gradient(
        zero[:4], (4,), (1.,))[0]
    calls = []

    def evaluate(params_list):
        calls.append(len(params_list))
        return [(np.zeros(4), jacobian) for _ in params_list]

    def update(p, f, mean, j, y, cd, covariance, damping):
        return tv.linearized_step(p, f, mean, j, y, cd, strength=.3, **kwargs)

    records = update_members_once([params, params], [datum, datum], params, np.eye(8),
                                  np.ones(4), evaluate, update,
                                  lambda f, y, *args: float(np.sum((f - y) ** 2)), active)
    assert calls == [2]
    assert all(record['prediction_kind'] == 'linearized' for record in records)
    assert np.max(abs(records[0]['parameters'])) <= 5.


@pytest.mark.parametrize('stage_weights', [None, (.2, .05)])
def test_controller_tv_dose_is_independent_of_iteration_count(tmp_path, monkeypatch, stage_weights):
    import h5py
    import pandas as pd
    import run_DA_sequential as seq

    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    depths = np.array([100., 101.524]) * seq.meter_to_feet
    with h5py.File(benchmark / 'globalmodel.h5', 'w') as handle:
        handle['wellpath/Z'] = np.array([[100.], [101.524]])
    tool = ('6kHz', '83ft')
    pd.DataFrame({tool: [np.ones(8)] * 2}, index=depths).to_pickle(tmp_path / 'data.pkl')
    pd.DataFrame({tool: [['ABS', [1.] * 8]] * 2}, index=depths).to_pickle(tmp_path / 'var.pkl')

    def setup(*args):
        monkeypatch.setattr(seq, 'TVD', depths, raising=False)
        monkeypatch.setattr(seq, 'WX', np.array([0., 5.]), raising=False)
        monkeypatch.setattr(seq, 'WY', np.zeros(2), raising=False)

    class Adapter:
        def backend(self):
            return self

        def trusts_ad(self, fidelity):
            return True

        def runtime_identity(self):
            return {}

    batches = []

    def gradient(params, index):
        batches.append((index, len(params)))
        return [(np.ones(8) * 5, np.zeros((8, len(p)))) for p in params]

    monkeypatch.setattr(seq, 'setup_simulators', setup)
    monkeypatch.setattr(seq, '_set_logging_point_state', lambda index: None)
    monkeypatch.setattr(seq, 'make_remote2d', lambda *args: Adapter())
    monkeypatch.setattr(seq, 'simulate_pred_only',
                        lambda p, index, fidelity: np.ones(8) * (1 if index == 0 else 5))
    monkeypatch.setattr(seq, 'simulate_and_grad_multi', gradient)
    output = tmp_path / 'run'
    seq.main(['--benchmark', str(benchmark), '--data', str(tmp_path / 'data.pkl'),
              '--variance', str(tmp_path / 'var.pkl'), '--output-dir', str(output),
              '--nx', '3', '--nz', '4', '--ensemble-size', '6', '--opt-members', '2',
              '--single-update', '--regularization', 'tv', '--tv-weight', '.1'] +
             (['--tv-weight-1d', '.2', '--tv-weight-2d', '.05'] if stage_weights else []))
    for index in range(2):
        result = pd.read_pickle(output / f'inversion_results_assim_{index}.pkl')
        if stage_weights is None:
            assert result['tv_strength_applied'] == pytest.approx(.1)
        else:
            if index == 0:
                assert result['tv_strength_applied'] == pytest.approx(.2)
            else:
                assert .05 < result['tv_strength_applied'] < .2
            for step in result['tv_history']:
                assert step['strength'] / step['assimilation_fraction'] == pytest.approx(
                    .05 if step['stage'] == '2D' else .2)
        assert sum(step['assimilation_fraction'] for step in result['tv_history']) == pytest.approx(1.)
        assert result['prior_projection_effective'] is False
    assert batches == [(1, 2)]
