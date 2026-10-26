import numpy as np

from rolling_prior import RollingPriorBuffer


def make_buffer(members=12, seed=17):
    return RollingPriorBuffer.create(
        (9, 7), (1., 1.), ((3., 2.), (4., 3.)), (.8, 0.), (.5, .12),
        members, 4., seed)


def test_overlap_and_prior_halo_are_independent_of_posterior_updates():
    first = make_buffer()
    second = make_buffer()
    before = first.active_fields()
    posterior = before.copy()
    posterior += 10.
    moved = first.restart(posterior.reshape(126, -1), (2, -1)).reshape(before.shape)
    second.advance((2, -1))
    np.testing.assert_array_equal(first.fields, second.fields)
    np.testing.assert_array_equal(moved[:, :7, 1:], posterior[:, 2:, :6])
    np.testing.assert_array_equal(moved[:, 7:], second.active_fields()[:, 7:])
    np.testing.assert_array_equal(moved[:, :, 0], second.active_fields()[:, :, 0])


def test_checkpoint_restores_prior_and_random_stream(tmp_path):
    buffer = make_buffer()
    buffer.advance((3, 1))
    path = tmp_path / 'prior-buffer.npz'
    buffer.save(path)
    resumed = RollingPriorBuffer.load(path)
    for movement in ((1, 2), (-2, -1), (30, 0)):
        buffer.advance(movement)
        resumed.advance(movement)
        np.testing.assert_array_equal(buffer.fields, resumed.fields)
        assert buffer.origin == resumed.origin


def test_no_data_many_shifts_preserve_prior_variance_and_covariance():
    buffer = make_buffer(members=5000)
    for _ in range(25):
        buffer.advance((1, 1))
    fields = buffer.active_fields()
    for field, (mean, std) in enumerate(((.8, .5), (0., .12))):
        normalized = (fields[field] - mean) / std
        np.testing.assert_allclose(normalized.mean(axis=2), 0., atol=.06)
        np.testing.assert_allclose(normalized.var(axis=2), 1., atol=.10)
        for distance, expected in ((1, np.exp(-1 / 9)), (3, np.exp(-1.))):
            if field == 1:
                expected = np.exp(-distance / 4.)
            covariance = np.mean(normalized[:-distance] * normalized[distance:], axis=2)
            np.testing.assert_allclose(covariance, expected, atol=.08)
        expected_vertical = np.exp(-1 / 4) if field == 0 else np.exp(-1 / 3)
        covariance = np.mean(normalized[:, :-1] * normalized[:, 1:], axis=2)
        np.testing.assert_allclose(covariance, expected_vertical, atol=.08)


def test_zero_shift_retains_every_posterior_cell():
    buffer = make_buffer()
    posterior = np.full((126, 12), 7.)
    np.testing.assert_array_equal(buffer.restart(posterior, (0, 0)), posterior)
