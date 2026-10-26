import numpy as np
import pytest

import pilot_rml_index45 as rml
from sparse_precision import EnsemblePrecisionPrior


def study(counts=None):
    base = {'dims': [3, 5, 4], 'spacing_ft': [5., 5., 5.]}
    return base if counts is None else {**base, 'graded_dz': counts}


def test_uniform_grid_maps_are_identities():
    expand, contract, dims, spacing = rml.grid_maps(study())
    raw = np.arange(24.)
    np.testing.assert_array_equal(expand(raw), raw)
    assert dims == (3, 5, 4) and spacing == (5., 5., 5.)
    jacobian = np.ones((2, 24))
    assert contract(jacobian) is jacobian


def test_graded_expand_and_contract_are_adjoint():
    counts = [3, 1, 1, 5]
    expand, contract, dims, spacing = rml.grid_maps(study(counts))
    assert dims == (3, 5, 10) and spacing == (5., 5., 1.)
    raw = np.random.default_rng(0).normal(size=24)
    fine = expand(raw)
    assert fine.shape == (60,)
    # Row-wise repetition within every column, both blocks.
    np.testing.assert_array_equal(fine.reshape(2, 3, 10)[1, 2], np.repeat(raw.reshape(2, 3, 4)[1, 2], counts))
    jacobian = np.random.default_rng(1).normal(size=(7, 60))
    np.testing.assert_allclose(contract(jacobian) @ raw, jacobian @ fine)


def test_graded_grid_rejects_bad_counts():
    with pytest.raises(ValueError):
        rml.grid_maps(study([1, 2, 3]))
    with pytest.raises(ValueError):
        rml.grid_maps(study([1, 0, 2, 3]))


def test_scaled_prior_widens_every_std():
    ensemble = np.random.default_rng(2).normal(size=(2 * 3 * 4, 40))
    base = EnsemblePrecisionPrior.fit(ensemble, (3, 4), 1, (.01, .01))
    scaled = rml.ScaledPrior(base, 10.)
    np.testing.assert_allclose(scaled.std, 10 * base.std)
    z = np.random.default_rng(3).normal(size=24)
    np.testing.assert_allclose(scaled.whiten(scaled.color(z)), z, atol=1e-10)
    offset = scaled.color(z)
    assert np.isclose(.5 * scaled.whiten(offset) @ scaled.whiten(offset), .5 * z @ z)
    np.testing.assert_allclose(offset, 10 * base.color(z))
    with pytest.raises(ValueError):
        rml.ScaledPrior(base, 0.)


def test_load_prior_applies_the_study_scale(tmp_path):
    ensemble = np.random.default_rng(4).normal(size=(24, 40))
    base = EnsemblePrecisionPrior.fit(ensemble, (3, 4), 1, (.01, .01))
    base.save(tmp_path / 'prior-precision.npz')
    spec = {'type': 'ensemble_precision', 'file': 'prior-precision.npz'}
    plain = rml.load_prior({'prior': spec}, tmp_path)
    scaled = rml.load_prior({'prior': {**spec, 'scale': 4.}}, tmp_path)
    assert not isinstance(plain, rml.ScaledPrior)
    np.testing.assert_allclose(scaled.std, 4 * plain.std)


def test_graded_rows_keep_thickness_and_count():
    import multistart_point as ms
    counts = ms.graded_rows(400, 80, 204)
    assert sum(counts) == 400 and len(counts) == 80
    tool = sum(counts[:counts.index(1)])
    above, below = counts[:40], counts[41:]
    assert counts[40] == 1 and sum(counts[:40]) == 204
    assert all(a >= b for a, b in zip(above, above[1:])) and all(a <= b for a, b in zip(below, below[1:]))
    matrix = ms.resample_rows([5] * 80, counts)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.)
    np.testing.assert_allclose(ms.resample_state(np.ones(2 * 3 * 80), 3, matrix), 1.)


def test_directional_floor_only_touches_usdp_and_usda():
    from sequential_matern_1d import directional_floor
    from esmda_remote2d import UDAR_COMPONENTS
    covariance = np.full(16, 1e-6)
    floored = directional_floor(covariance, (.5, .25))
    for channel, value in enumerate(floored):
        name = UDAR_COMPONENTS[channel % 8]
        expected = {'USDP': .25, 'USDA': .0625}.get(name, 1e-6)
        assert value == pytest.approx(expected)
    big = directional_floor(np.full(8, 4.), (.5, .25))
    np.testing.assert_allclose(big, 4.)                      # sigma above the floor is kept
    with pytest.raises(ValueError):
        directional_floor(np.ones(8), (.5, 0.))


def test_loop_passes_the_directional_floor_on_resume():
    import run_chain_laplace_loop as loop
    study = {'ensemble_size': 250, 'seed': 1, 'mda_steps': 10, 'tempering': 'linear-inflation',
             'anisotropy': {'parameterization': 'log_clipped', 'state_moments': [.5, .5]},
             'directional_sigma_floor': [.5, .25]}
    arguments = loop.driver_arguments(study, 161)
    position = arguments.index('--directional-sigma-floor')
    assert arguments[position + 1:position + 3] == ['0.5', '0.25']
    assert '--directional-sigma-floor' not in loop.driver_arguments({**study, 'directional_sigma_floor': None}, 161)
