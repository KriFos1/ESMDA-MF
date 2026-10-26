import numpy as np
import pytest

from esmda_core import measurement_misfit
from pilot_cascade_laplace_index34 import (MAP_INFLATION, cascade_inflation, draw_prior,
                                           original_background_prior, restart_ensemble,
                                           select_best_member)
import run_DA_sequential as sequential


MANIFEST = {'dims': [3, 1, 6], 'dx_ft': 5., 'dz_ft': 5.,
            'prior_background': np.log10(1.5), 'prior_reservoir': np.log10(30.),
            'prior_mode_std': .1, 'prior_reservoir_fraction': .15,
            'prior_switch_prob': .05, 'prior_latent_correlation_x_ft': 80.,
            'prior_latent_correlation_z_ft': 35.,
            'rv_parameterization': {'latent_std': .12}}


def test_restart_ensemble_reindexes_columns_and_caps_entered_cells():
    sequential.dims = (3, 1, 6)
    rng = np.random.default_rng(77)
    cells = 18
    state = rng.normal(.5, .05, size=(2 * cells, 5))
    state[:cells] = np.clip(state[:cells], -1., np.log(40.) - .01)
    shifted = restart_ensemble(state, (1, 0), MANIFEST)
    assert shifted.shape == state.shape
    old = state[:cells].reshape(3, 6, 5)
    new = shifted[:cells].reshape(3, 6, 5)
    np.testing.assert_allclose(new[0], old[1])
    np.testing.assert_allclose(new[1], old[2])
    assert not np.allclose(new[2], old[2])
    assert np.isfinite(shifted).all()
    assert shifted[:cells].max() <= np.log(40.) + 1e-12


def test_restart_ensemble_extends_rows_for_vertical_movement():
    sequential.dims = (3, 1, 6)
    rng = np.random.default_rng(78)
    cells = 18
    state = rng.normal(.5, .05, size=(2 * cells, 5))
    state[:cells] = np.clip(state[:cells], -1., np.log(40.) - .01)
    shifted = restart_ensemble(state, (0, 1), MANIFEST)
    old = state[cells:].reshape(3, 6, 5)
    new = shifted[cells:].reshape(3, 6, 5)
    np.testing.assert_allclose(new[:, 0], old[:, 1])
    np.testing.assert_allclose(new[:, 1], old[:, 2])
    assert not np.allclose(new[:, 5], old[:, 5])


def test_mda_schedules_spend_half_the_likelihood_weight():
    for steps in (1, 4, 5, 10):
        schedule = cascade_inflation(steps)
        assert schedule.shape == (steps,)
        assert np.all(np.diff(schedule) < 0)
        np.testing.assert_allclose(sum(1 / alpha for alpha in schedule), .5, atol=1e-12)
    standard = np.arange(15., 5., -1.) * np.reciprocal(np.arange(15., 5., -1.)).sum()
    np.testing.assert_allclose(cascade_inflation(10), 2. * standard, rtol=1e-12)
    assert MAP_INFLATION == pytest.approx(2.)


def test_five_equal_energy_steps_use_tenfold_covariance():
    alpha = cascade_inflation(5, 'equal-energy')
    np.testing.assert_array_equal(alpha, np.full(5, 10.))
    np.testing.assert_allclose(np.cumsum(1. / alpha), [.1, .2, .3, .4, .5])
    assert cascade_inflation(5, 'linear-inflation')[0] != 10.
    with pytest.raises(ValueError):
        cascade_inflation(5, 'unknown')


def test_full_energy_ten_step_linear_mda():
    alpha = cascade_inflation(10, 'linear-inflation', likelihood_weight=1.)
    np.testing.assert_allclose(np.sum(1 / alpha), 1., atol=1e-14)
    assert np.all(np.diff(alpha) < 0)
    np.testing.assert_allclose(np.diff(alpha), np.diff(alpha)[0])
    np.testing.assert_allclose(alpha, cascade_inflation(10) / 2.)


def test_background_prior_has_original_stationary_moments_and_kernels():
    prior = original_background_prior(MANIFEST)
    mean_rh = .85 * np.log(1.5) + .15 * np.log(30.)
    variance_rh = (.1 * np.log(10.)) ** 2 + .15 * .85 * np.log(20.) ** 2
    np.testing.assert_allclose(prior.mean[:18], mean_rh)
    np.testing.assert_array_equal(prior.mean[18:], 0.)
    np.testing.assert_allclose(prior.std[:18] ** 2, variance_rh)
    np.testing.assert_allclose(prior.std[18:], .12)
    covariance = prior.apply(np.eye(36))
    assert covariance[0, 1] / variance_rh == pytest.approx(np.exp(-.1 ** 2))
    assert covariance[18, 19] / .12 ** 2 == pytest.approx(np.exp(-5 / 35))
    np.testing.assert_array_equal(covariance[:18, 18:], 0.)
    draw = prior.draw(100, np.random.default_rng(7))
    penalties = np.array([prior.quadratic(draw[:, j]) for j in range(100)])
    assert np.isfinite(penalties).all() and (penalties >= 0).all()
    assert penalties.mean() == pytest.approx(36., rel=.12)


def test_background_laplace_draws_match_untempered_linear_posterior():
    from lm_laplace import sample_laplace

    prior = original_background_prior(MANIFEST)
    jacobian = np.zeros((2, 36))
    jacobian[0, 6], jacobian[1, 24] = 1., 1.
    variance = np.array([.1, .03])
    solution = {'model': prior.mean + .2, 'jacobian': jacobian}
    draws = sample_laplace(solution, prior, variance, 5000, np.random.default_rng(9))
    covariance = prior.apply(np.eye(36))
    cross = covariance @ jacobian.T
    expected = covariance - cross @ np.linalg.solve(jacobian @ cross + np.diag(variance), cross.T)
    np.testing.assert_allclose(draws.var(axis=1, ddof=1), expected.diagonal(), rtol=.07)
    np.testing.assert_allclose(draws.mean(axis=1), solution['model'], atol=.025)


def test_best_member_selection_uses_original_covariance_misfit():
    predictions = np.array([[1., 3., 2.], [1., 3., 2.]])
    observations = np.array([2., 2.])
    covariance = np.array([1., 4.])
    best, misfits = select_best_member(predictions, observations, covariance)
    assert best == 2
    np.testing.assert_allclose(misfits, [
        measurement_misfit(predictions[:, j], observations, covariance)
        for j in range(3)])


def test_prior_draw_matches_reference_recipe_and_caps():
    previous = np.random.get_state()
    try:
        state = draw_prior(MANIFEST, 2031, 4)
    finally:
        np.random.set_state(previous)
    assert state.shape == (36, 4)
    assert np.isfinite(state).all()
    assert state[:18].max() <= np.log(40.) + 1e-12
    assert state[:18].min() >= -1. - 1e-12
