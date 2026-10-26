import numpy as np

from rml import gaussian_moment_redraw, run_rml


class DiagonalPrior:
    def __init__(self):
        self.mean = np.zeros(4)
        self.std = np.array([.4, .5, .3, .4])

    def apply(self, values):
        return self.std[:, None] ** 2 * values

    def solve(self, values):
        return values / self.std ** 2

    def draw(self, count, rng):
        return self.std[:, None] * rng.normal(size=(4, count))


def test_independent_rml_matches_each_untempered_linear_gaussian_minimizer():
    rng = np.random.default_rng(12)
    ensemble = rng.normal(.1, .1, (4, 250))
    J = np.array([[1., .2, .1, 0.], [0., .3, 0., 1.]])
    predictions = J @ ensemble
    observations = np.array([.2, -.1])
    variance = np.array([.02, .03])
    prior = DiagonalPrior()
    batch_sizes = []
    def forward(models):
        batch_sizes.append(len(models))
        return [J @ model for model in models]
    def derivative(models):
        batch_sizes.append(len(models))
        return [J for model in models]
    result = run_rml(ensemble, predictions, observations, variance, prior,
                     forward, derivative, np.random.default_rng(42),
                     max_iterations=15, max_trials=5, max_step=10.,
                     relative_data_improvement=1e-6, rh_limits=(-10., 10.))
    score = np.mean((predictions - observations[:, None]) ** 2 / variance[:, None], axis=0)
    np.testing.assert_array_equal(result['selected'], np.argsort(score, kind='stable')[:20])
    np.testing.assert_array_equal(result['initials'], ensemble[:, result['selected']])
    np.testing.assert_allclose(result['targets'], observations[:, None] + result['measurement_noise'])
    C = np.diag(prior.std ** 2)
    expected = result['prior_centers'] + C @ J.T @ np.linalg.solve(
        J @ C @ J.T + np.diag(variance), result['targets'] - J @ result['prior_centers'])
    fitted = np.column_stack([fit['model'] for fit in result['fits']])
    np.testing.assert_allclose(fitted, expected, atol=3e-5)
    assert result['usable'].all()
    assert batch_sizes[:3] == [20, 20, 20]
    assert np.ptp(result['prior_centers'], axis=1).min() > .1
    assert np.ptp(result['targets'], axis=1).min() > .1


def test_redraw_preserves_rml_marginal_mean_and_variance_before_bounds():
    rng = np.random.default_rng(7)
    models = rng.normal(size=(96, 20)) * np.linspace(.01, .3, 96)[:, None]
    models += np.linspace(0., 1., 96)[:, None]
    models[0] = .4
    raw, mean, std = gaussian_moment_redraw(
        models, (8, 6), (5., 5.), ((50., 50.), (80., 35.)), 250, 21)
    assert raw.shape == (96, 250)
    np.testing.assert_allclose(raw.mean(axis=1), models.mean(axis=1), atol=1e-14)
    np.testing.assert_allclose(raw.var(axis=1, ddof=1), models.var(axis=1, ddof=1), atol=1e-14)
    np.testing.assert_array_equal(raw[0], np.full(250, .4))
    np.testing.assert_allclose(mean, models.mean(axis=1))
    np.testing.assert_allclose(std, models.std(axis=1, ddof=1), atol=1e-14)
    assert np.corrcoef(raw[10], raw[16])[0, 1] > .9
