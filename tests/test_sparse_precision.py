import numpy as np
import pytest

from matern_prior import MaternPrior
from sparse_precision import EnsemblePrecisionPrior, predecessors, ridge_gcv

NX, NZ = 10, 12


@pytest.fixture(scope='module')
def matern():
    prior = MaternPrior((NX, NZ), (5., 5.), [[20., 25.], [30., 15.]], [1., .5], [.8, .5])
    root = prior.color(np.eye(len(prior.mean)))
    return prior, root @ root.T


def covariance(prior):
    root = prior.color(np.eye(len(prior.mean)))
    return root @ root.T


def test_ridge_gcv_recovers_coefficients_and_noise():
    rng = np.random.default_rng(0)
    predictors = rng.normal(size=(250, 10))
    beta = rng.normal(size=10)
    target = predictors @ beta + .5 * rng.normal(size=250)
    predictors -= predictors.mean(axis=0)
    target -= target.mean()
    coefficients, variance, _ = ridge_gcv(predictors, target)
    assert np.abs(coefficients - beta).max() < .15
    assert variance == pytest.approx(.25, rel=.2)
    # Unrelated predictors are shrunk away and the target variance is kept.
    noise = rng.normal(size=250)
    coefficients, variance, _ = ridge_gcv(predictors, noise - noise.mean())
    assert np.abs(coefficients).max() < .1 and variance == pytest.approx(np.var(noise, ddof=1), rel=.1)


def test_predecessors_are_earlier_neighbours_of_both_blocks():
    order, earlier = predecessors(NX, NZ, 2)
    cells = NX * NZ
    assert sorted(order) == list(range(2 * cells)) and order[0] == 0 and order[1] == cells
    position = {state: p for p, state in enumerate(order)}
    interior = position[cells + 5 * NZ + 6]          # latent of cell (5, 6)
    assert len(earlier[interior]) == 25              # 12 earlier cells x 2 blocks + its own ln Rh
    assert all(p < interior for p in earlier[interior])
    for p in earlier[interior]:
        x, z = divmod(order[p] % cells, NZ)
        assert abs(x - 5) <= 2 and abs(z - 6) <= 2


def test_fit_recovers_a_matern_covariance_better_than_the_sample_covariance(matern):
    prior, truth = matern
    scale = np.linalg.norm(truth)
    for members, tolerance in ((250, .35), (4000, .1)):
        ensemble = prior.sample(members, 1)
        estimate = covariance(EnsemblePrecisionPrior.fit(ensemble, (NX, NZ), radius=2, floors=(0., 0.)))
        error = np.linalg.norm(estimate - truth) / scale
        assert error < tolerance
        assert error < np.linalg.norm(np.cov(ensemble) - truth) / scale


def test_factor_operations_are_consistent(matern):
    prior, _ = matern
    fitted = EnsemblePrecisionPrior.fit(prior.sample(250, 2), (NX, NZ), radius=2)
    rng = np.random.default_rng(3)
    z, v = rng.normal(size=(len(fitted.mean), 3)), rng.normal(size=(len(fitted.mean), 3))
    np.testing.assert_allclose(fitted.whiten(fitted.color(z)), z, atol=1e-10)
    assert np.sum(fitted.color(z) * v) == pytest.approx(np.sum(z * fitted.color(v, transpose=True)))
    estimate = covariance(fitted)
    np.testing.assert_allclose(fitted.marginal_variance(), np.diag(estimate), rtol=1e-10)
    np.testing.assert_allclose(fitted.std, np.sqrt(np.diag(estimate)), rtol=1e-10)
    np.testing.assert_allclose(fitted.precision().toarray() @ estimate, np.eye(len(fitted.mean)), atol=1e-8)
    np.testing.assert_allclose(fitted.color(z[:, 0]), fitted.color(z)[:, 0])
    samples = fitted.sample(20000, 4)
    np.testing.assert_allclose(samples.std(axis=1), fitted.std, rtol=.05)


def test_floors_lift_small_spread_and_keep_correlations(matern):
    prior, _ = matern
    ensemble = prior.mean[:, None] + .05 * (prior.sample(250, 5) - prior.mean[:, None])
    floored = EnsemblePrecisionPrior.fit(ensemble, (NX, NZ), radius=2)
    plain = EnsemblePrecisionPrior.fit(ensemble, (NX, NZ), radius=2, floors=(0., 0.))
    cells = NX * NZ
    assert floored.std[:cells].min() >= .23 - 1e-12 and floored.std[cells:].min() >= .1 - 1e-12
    assert plain.std.max() < .1

    def correlation(matrix):
        std = np.sqrt(np.diag(matrix))
        return matrix / std[:, None] / std[None, :]
    np.testing.assert_allclose(correlation(covariance(floored)), correlation(covariance(plain)), atol=1e-10)
    np.testing.assert_allclose(floored.mean, ensemble.mean(axis=1))


def test_save_and_load_reproduce_the_prior(tmp_path, matern):
    prior, _ = matern
    fitted = EnsemblePrecisionPrior.fit(prior.sample(250, 6), (NX, NZ), radius=2)
    fitted.save(tmp_path / 'prior.npz')
    loaded = EnsemblePrecisionPrior.load(tmp_path / 'prior.npz')
    z = np.random.default_rng(7).normal(size=len(fitted.mean))
    np.testing.assert_array_equal(loaded.color(z), fitted.color(z))
    np.testing.assert_array_equal(loaded.std, fitted.std)
    assert loaded.shape == (2, NX, NZ) and loaded.summary() == fitted.summary()
    with pytest.raises(FileExistsError):
        fitted.save(tmp_path / 'prior.npz')
    with pytest.raises(ValueError):
        EnsemblePrecisionPrior.fit(prior.sample(250, 6)[:-1], (NX, NZ))


def test_conditional_regression_is_the_gaussian_conditional(matern):
    from sparse_precision import conditional_regression
    prior, truth = matern
    precision = np.linalg.inv(truth)
    observed = np.r_[np.arange(5 * NZ, 6 * NZ), NX * NZ + np.arange(5 * NZ, 6 * NZ)]   # one column, both blocks
    rest, propagate = conditional_regression(precision, observed)
    assert len(rest) + len(observed) == 2 * NX * NZ and not set(rest) & set(observed)
    increment = np.random.default_rng(9).normal(size=(len(observed), 4))
    expected = truth[np.ix_(rest, observed)] @ np.linalg.solve(truth[np.ix_(observed, observed)], increment)
    np.testing.assert_allclose(propagate(increment), expected, atol=1e-8)
    np.testing.assert_allclose(propagate(increment[:, 0]), expected[:, 0], atol=1e-8)
    # Assimilating data that see only the observed block leaves the regression unchanged.
    sensitivity = np.random.default_rng(10).normal(size=(3, len(observed)))
    cross = truth[:, observed] @ sensitivity.T
    posterior = truth - cross @ np.linalg.solve(sensitivity @ truth[np.ix_(observed, observed)] @ sensitivity.T
                                                + .1 * np.eye(3), cross.T)
    _, updated = conditional_regression(np.linalg.inv(posterior), observed)
    np.testing.assert_allclose(updated(increment), expected, atol=1e-6)
