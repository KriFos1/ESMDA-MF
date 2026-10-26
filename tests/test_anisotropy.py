import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import norm

from esmda_core import (latent_slope, log_ratio_from_latent, log_rv_from_latent,
                        optimizer_jacobian)


def test_log_clipped_reaches_isotropy_at_finite_values_and_is_linear_above_zero():
    latent = np.array([-2., -.1, 0., .3, .981])
    np.testing.assert_array_equal(log_ratio_from_latent(latent, anisotropy='log_clipped'),
                                  [0., 0., 0., .3, .981])
    np.testing.assert_array_equal(latent_slope(latent, anisotropy='log_clipped'), [0, 0, 0, 1, 1])
    np.testing.assert_allclose(log_ratio_from_latent(latent, 1., 3.),
                               np.log(3.) * norm.cdf(latent))
    with pytest.raises(ValueError):
        log_ratio_from_latent(latent, anisotropy='bogus')


@pytest.mark.parametrize('anisotropy', ['probit', 'log_clipped'])
def test_optimizer_jacobian_matches_finite_differences_with_rv_cap(anisotropy):
    rng = np.random.default_rng(4)
    cells = 6
    j_rh, j_rv = rng.normal(size=(3, cells)), rng.normal(size=(3, cells))
    log_rh = np.array([.4, 1., 2.5, 3.5, .8, 3.2])
    latent = np.array([.6, -.4, .9, .5, 1.3, .2])          # away from kinks at 0 and the cap
    cap = np.log(40.)

    def response(state):
        rh, lat = state[:cells], state[cells:]
        return j_rh @ rh + j_rv @ log_rv_from_latent(rh, lat, 1., 3., cap, anisotropy)
    state = np.r_[log_rh, latent]
    analytic = optimizer_jacobian(j_rh, j_rv, latent, 1., 3., log_rv_cap=cap, log_rh=log_rh,
                                  anisotropy=anisotropy)
    np.testing.assert_allclose(analytic, approx_derivative(response, state, method='3-point'),
                               atol=1e-7)


def test_sequential_map_uses_the_selected_parameterization(monkeypatch):
    import run_DA_sequential as sequential
    monkeypatch.setattr(sequential, 'dims', (2, 1, 2))
    monkeypatch.setattr(sequential, 'log_rv_cap', np.log(40.))
    param = np.r_[np.log([1.5, 20., 30., 1.5]), [.981, -.5, 0., .981]]
    monkeypatch.setattr(sequential, 'anisotropy', 'log_clipped')
    parameters, log_rh, _ = sequential._build_param_dict(param)
    np.testing.assert_allclose(np.exp(parameters['rv'][:4]), [4.0, 30., 20., 4.0], rtol=1e-3)
    monkeypatch.setattr(sequential, 'anisotropy', 'probit')
    probit = sequential._build_param_dict(param)[0]['rv']
    assert not np.allclose(probit, parameters['rv'])


def test_cascade_matern_draws_and_diagnostics(monkeypatch):
    import pilot_cascade_laplace_index34 as cascade
    import run_DA_sequential as sequential
    manifest = {'dims': [12, 5, 16], 'dx_ft': 5., 'dz_ft': 5., 'prior_background': np.log10(1.5),
                'prior_reservoir': np.log10(30.), 'prior_mode_std': .1, 'prior_reservoir_fraction': .15,
                'prior_latent_correlation_x_ft': 80., 'prior_latent_correlation_z_ft': 35.,
                'rv_parameterization': {'latent_std': .12}}
    state, prior = cascade.draw_matern_prior(manifest, 3, 40, 'log_clipped', .5, .5)
    cells = 12 * 16
    assert state.shape == (2 * cells, 40) and np.all(state[:cells] <= np.log(40.) + 1e-12)
    assert abs(state[cells:].mean() - .5) < .15 and abs(state[cells:].std() - .5) < .1
    monkeypatch.setattr(sequential, 'max_ratio', 3.)
    diagnostics = cascade.ensemble_diagnostics(state, 12, 16, prior, 'log_clipped')
    assert 0 < diagnostics['isotropic_fraction'] < .5
    assert .5 < diagnostics['matern_2p_over_n_latent_median'] < 2.


def test_rml_study_prior_reproduces_the_cascade_matern_prior(tmp_path):
    import json
    import pilot_cascade_laplace_index34 as cascade
    from matern_prior import from_study_prior
    from pilot_rml_index45 import cascade_matern_prior
    manifest = {'dims': [12, 5, 16], 'dx_ft': 5., 'dz_ft': 5., 'prior_background': np.log10(1.5),
                'prior_reservoir': np.log10(30.), 'prior_mode_std': .1, 'prior_reservoir_fraction': .15,
                'prior_latent_correlation_x_ft': 80., 'prior_latent_correlation_z_ft': 35.,
                'rv_parameterization': {'latent_std': .12}}
    (tmp_path / 'run.json').write_text(json.dumps(manifest))
    study = {'prior_family': 'matern', 'reference': str(tmp_path),
             'anisotropy': {'parameterization': 'log_clipped', 'state_moments': [.5, .5]}}
    prior, anisotropy, _ = cascade_matern_prior(study)
    state, _ = cascade.draw_matern_prior(manifest, 7, 5, 'log_clipped', .5, .5)
    redrawn = cascade.cap_rh(cascade.clamp_log_rh(from_study_prior(prior).sample(5, 7)), 40.)
    assert anisotropy == 'log_clipped' and prior['type'] == 'matern'
    np.testing.assert_array_equal(redrawn, state)
    with pytest.raises(ValueError):
        cascade_matern_prior({**study, 'prior_family': 'fast_gaussian'})
