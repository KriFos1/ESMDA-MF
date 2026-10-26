import json

import numpy as np

import pilot_2d_only_index31 as pilot
from pilot_2d_only_index31 import draw_prior


def test_prior_draw_is_fresh_reproducible_and_spatially_variable():
    prior = {'prior_style': 'gaussian', 'prior_projection': 'off',
             'dims': [3, 1, 6], 'dx_ft': 5., 'dz_ft': 5.,
             'prior_background': np.log10(1.5), 'prior_reservoir': np.log10(30.),
             'prior_mode_std': .1, 'prior_reservoir_fraction': .15,
             'prior_switch_prob': .05, 'prior_correlation_x_ft': 10.,
             'prior_correlation_z_ft': 75.,
             'prior_latent_correlation_z_ft': 35.,
             'prior_latent_correlation_x_ft': 80.,
             'rv_parameterization': {'latent_std': .12}}
    previous = np.random.get_state()
    try:
        first = draw_prior(prior, 2031)
        same_seed = draw_prior(prior, 2031)
        new_seed = draw_prior(prior, 2032)
    finally:
        np.random.set_state(previous)
    assert first.shape == (36, 25)
    np.testing.assert_array_equal(first, same_seed)
    assert not np.array_equal(first, new_seed)
    assert np.std(first[:18], axis=1).min() > 0
    assert np.std(first[:18, 0]) > 0


def test_correlated_prior_preserves_rh_draw_and_latent_scale():
    prior = {'prior_style': 'gaussian', 'prior_projection': 'off',
             'dims': [33, 5, 80], 'dx_ft': 5., 'dz_ft': 5.,
             'prior_background': np.log10(1.5), 'prior_reservoir': np.log10(30.),
             'prior_mode_std': .1, 'prior_reservoir_fraction': .15,
             'prior_switch_prob': .05, 'prior_correlation_x_ft': 10.,
             'prior_correlation_z_ft': 75.,
             'prior_latent_correlation_z_ft': 35.,
             'prior_latent_correlation_x_ft': 80.,
             'rv_parameterization': {'latent_std': .12}}
    previous = np.random.get_state()
    try:
        correlated = draw_prior(prior, 2031, correlation=.8)
        independent = draw_prior(prior, 2031)
    finally:
        np.random.set_state(previous)
    cells = 33 * 80
    np.testing.assert_array_equal(correlated[:cells], independent[:cells])
    empirical = np.corrcoef(correlated[:cells].ravel(), correlated[cells:].ravel())[0, 1]
    assert abs(empirical - .8) < .08
    uncoupled = np.corrcoef(independent[:cells].ravel(), independent[cells:].ravel())[0, 1]
    assert abs(uncoupled) < .08
    assert abs(np.std(correlated[cells:]) - .12) < .03
    assert abs(np.mean(correlated[cells:])) < .03
    rh = independent[:cells].reshape(33, 80, 25)
    lag_correlation = lambda left, right: np.corrcoef(left.ravel(), right.ravel())[0, 1]
    assert lag_correlation(rh[:-1], rh[1:]) > .95
    assert lag_correlation(rh[:, :-1], rh[:, 1:]) > .95
    assert abs(lag_correlation(rh[:-10], rh[10:]) - np.exp(-1.)) < .15
    assert abs(lag_correlation(rh[:, :-10], rh[:, 10:]) - np.exp(-1.)) < .15


def test_fast_gaussian_configured_moments_match_previous_sampler():
    prior = {'prior_background': np.log10(1.5), 'prior_reservoir': np.log10(30.),
             'prior_mode_std': .1, 'prior_reservoir_fraction': .15}
    mean, std = pilot.rh_prior_moments(prior)
    old_mean, old_covariance = pilot.sequential.mixture_stats(
        1, prior['prior_background'] * np.log(10.),
        prior['prior_reservoir'] * np.log(10.),
        prior['prior_mode_std'] * np.log(10.), prior['prior_reservoir_fraction'])
    np.testing.assert_allclose(mean, old_mean[0])
    np.testing.assert_allclose(std ** 2, old_covariance[0, 0])


def test_failed_numpy_member_ids_are_serializable(monkeypatch):
    monkeypatch.setattr(pilot, 'rh_rmse', lambda *args: .5)
    record = pilot.summarize_step(
        0, np.array([2, 5, 7]), np.array([2, 7]),
        np.array([[1., 2.], [3., 4.]]), np.array([1., 3.]),
        np.ones(2), np.ones((4, 2)), {}, None, 10.)
    assert json.loads(json.dumps(record))['failed_members'] == [5]


def test_two_d_forward_caps_vertical_resistivity_without_changing_rh(monkeypatch):
    monkeypatch.setattr(pilot.sequential, 'dims', (1, 5, 2))
    monkeypatch.setattr(pilot.sequential, 'log_rv_cap', np.log(50.))
    parameter = np.array([np.log(40.), np.log(45.), 10., 10.])
    fields, _, _ = pilot.sequential._build_param_dict(parameter)
    np.testing.assert_allclose(np.exp(fields['rh']).max(), 45.)
    np.testing.assert_allclose(np.exp(fields['rv']), 50.)


def test_default_caps_keep_rh_rv_below_40_and_vertical_above_horizontal(monkeypatch):
    monkeypatch.setattr(pilot.sequential, 'dims', (1, 5, 2))
    monkeypatch.setattr(pilot.sequential, 'log_rv_cap', np.log(pilot.RV_CAP_OHMM))
    parameter = np.array([4., np.log(35.), 10., -10.])
    original_latent = parameter[2:].copy()
    pilot.cap_rh(parameter, pilot.RH_CAP_OHMM)
    np.testing.assert_array_equal(parameter[2:], original_latent)
    fields, _, _ = pilot.sequential._build_param_dict(parameter)
    rh, rv = np.exp(fields['rh']), np.exp(fields['rv'])
    assert rh.max() <= 40. + 1e-10
    assert rv.max() <= 40. + 1e-10
    assert np.all(rv >= rh - 1e-10)
