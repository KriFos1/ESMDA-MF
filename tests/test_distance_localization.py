import numpy as np
import pytest

from distance_localization import (FurrerBengtssonLocalization, blend_localized_samples,
                                   localized_rtps)
from lm_laplace import SpatialPrior, optimize
from rolling_prior import RollingPriorBuffer
from transition_mda_study import update, updater_for


def make_localization(tmp_path, members=100, channels=2):
    return FurrerBengtssonLocalization((33, 80), (5., 5.), (16, 40), (60., 150.),
                                      members, channels, 8, tmp_path / 'localization.pkl')


def test_pet_fb_mask_physical_ranges_and_orientation(tmp_path):
    loc = make_localization(tmp_path)
    weights = loc.weights.reshape(2, 33, 80)
    np.testing.assert_array_equal(weights[0], weights[1])
    assert weights[0, 16, 40] == pytest.approx(100 / 102)
    assert weights[0, 27, 40] > 0
    assert weights[0, 28, 40] == 0
    assert weights[0, 16, 69] > 0
    assert weights[0, 16, 70] == 0
    assert not weights[:, :5].any()
    assert not weights[:, -5:].any()
    assert not weights[:, :, :11].any()
    assert not weights[:, :, 70:].any()
    np.testing.assert_allclose(weights[0, 18, 40], weights[0, 16, 45])


def test_pet_distance_update_preserves_collar_and_updates_correlated_interior(tmp_path):
    loc = make_localization(tmp_path)
    rng = np.random.default_rng(11)
    latent = rng.normal(size=100)
    state = np.ones((5280, 100)) * (.5 + .05 * latent)
    predicted = np.vstack([latent, 2 * latent])
    updater = updater_for(state, 2640, None, loc)
    assert 'autoadaloc' not in updater.localization.loc_info
    assert 'dist_loc' in updater.keys_da['localization']
    result = update(state, predicted, np.array([2., 4.]), np.ones(2) * .1,
                    1., rng, 2640, None, loc)
    np.testing.assert_array_equal(result[loc.weights == 0], state[loc.weights == 0])
    center = 16 * 80 + 40
    assert result[center].mean() > state[center].mean()


def test_advancing_collar_remains_the_original_prior_after_updates(tmp_path):
    members = 60
    loc = make_localization(tmp_path, members=members, channels=1)
    prior = RollingPriorBuffer.create((33, 80), (5., 5.), ((50., 50.), (80., 35.)),
                                     (.5, 0.), (.1, .12), members, 150., 12)
    state = prior.active_fields().reshape(5280, members)
    rng = np.random.default_rng(15)
    for _ in range(12):
        prediction = state[[16 * 80 + 40]]
        state = update(state, prediction, np.array([.8]), np.array([.001]),
                       1., rng, 2640, None, loc)
        np.testing.assert_array_equal(state.reshape(2, 33, 80, members)[:, -5:],
                                      prior.active_fields()[:, -5:])
        state = prior.restart(state, (1, 0))


def test_map_and_laplace_proposal_cannot_change_protected_members():
    rng = np.random.default_rng(3)
    incoming = rng.normal(.5, .1, (30, 20))
    prior = SpatialPrior(incoming, (5, 3), (5., 5.), ((20., 20.), (20., 20.)))
    initial = incoming[:, 0]
    weights = np.zeros(30)
    weights[[7, 22]] = .8
    jacobian = rng.normal(size=(3, 30))
    def forward(model):
        np.testing.assert_array_equal(model[weights == 0], initial[weights == 0])
        return jacobian @ model
    result = optimize(initial, prior, np.ones(3), np.ones(3), forward,
                      lambda model: jacobian, max_iterations=3, step_weights=weights)
    assert len(result['objective_history']) > 1
    np.testing.assert_array_equal(result['model'][weights == 0], initial[weights == 0])
    proposals = rng.normal(size=incoming.shape)
    samples = blend_localized_samples(incoming, proposals, weights)
    np.testing.assert_array_equal(samples[weights == 0], incoming[weights == 0])


def test_ranges_that_reach_window_boundary_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='collar'):
        FurrerBengtssonLocalization((33, 80), (5., 5.), (16, 40), (80., 150.),
                                   100, 2, 8, tmp_path / 'localization.pkl')


def test_wider_ranges_leave_two_cell_widths_to_nearest_boundary_centers(tmp_path):
    loc = FurrerBengtssonLocalization((33, 80), (5., 5.), (16, 40), (70., 185.),
                                      100, 2, 20, tmp_path / 'localization.pkl')
    weights = loc.weights.reshape(2, 33, 80)
    assert weights[0, 29, 40] > 0
    assert weights[0, 30, 40] == 0
    assert weights[0, 16, 76] > 0
    assert weights[0, 16, 77] == 0
    assert 80 - 70 == 2 * 5
    assert 195 - 185 == 2 * 5
    assert not weights[:, -3:].any()
    assert not weights[:, :, -3:].any()


def test_rtps_restores_target_spread_preserving_mean_correlations_and_collar():
    rng = np.random.default_rng(29)
    ensemble = rng.normal(size=(4, 200)) * np.array([.1, .2, .3, .4])[:, None]
    ensemble += np.array([1., 2., 3., 4.])[:, None]
    before = ensemble.std(axis=1, ddof=1)
    reference = before * np.array([3., 2., .5, 4.])
    weights = np.array([1., .5, 1., 0.])
    result, diagnostics = localized_rtps(ensemble, reference, weights, .25)
    np.testing.assert_allclose(result.std(axis=1, ddof=1), before * [1.5, 1.125, 1., 1.])
    np.testing.assert_allclose(result.mean(axis=1), ensemble.mean(axis=1), atol=1e-14)
    np.testing.assert_allclose(np.corrcoef(result), np.corrcoef(ensemble), atol=1e-14)
    np.testing.assert_array_equal(result[2:], ensemble[2:])
    assert not diagnostics['zero_spread_unrestored'].any()
    unchanged, _ = localized_rtps(ensemble, reference, weights, 0.)
    np.testing.assert_array_equal(unchanged, ensemble)


def test_rtps_reports_zero_spread_instead_of_inventing_ensemble_directions():
    ensemble = np.ones((2, 10))
    result, diagnostics = localized_rtps(ensemble, np.ones(2), np.array([1., 0.]), .25)
    np.testing.assert_array_equal(result, ensemble)
    np.testing.assert_array_equal(diagnostics['zero_spread_unrestored'], [True, False])
    with pytest.raises(ValueError, match='RTPS'):
        localized_rtps(ensemble, np.ones(2), np.ones(2), 1.1)
