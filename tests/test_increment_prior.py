import numpy as np
import pytest

from increment_prior import (IncrementPrior, ensemble_student_weights, face_differences, increment_precision,
                             student_weights)

SHAPE = (4, 6)
SCALES = (.1, .03, .05, .015)


def test_sparse_precision_matches_the_dense_prior():
    rng = np.random.default_rng(1)
    cells = SHAPE[0] * SHAPE[1]
    vertical, lateral = rng.uniform(.5, 200., (2, 4, 5)), rng.uniform(.5, 200., (2, 3, 6))
    anchor = np.repeat([3., 1.5], cells)
    precision = increment_precision(SHAPE, vertical, lateral, (3., 1.5)).toarray()
    d_vertical, d_lateral = face_differences(SHAPE)
    for block in range(2):
        rows = slice(block * cells, (block + 1) * cells)
        dense = (d_vertical.T @ (vertical[block].ravel()[:, None] * d_vertical)
                 + d_lateral.T @ (lateral[block].ravel()[:, None] * d_lateral) + np.eye(cells) / anchor[rows] ** 2)
        np.testing.assert_allclose(precision[rows, rows], dense)
    assert not precision[:cells, cells:].any()                    # the blocks are independent
    prior = IncrementPrior(np.zeros(2 * cells), anchor, vertical, lateral, SHAPE)
    np.testing.assert_allclose(np.sqrt(np.diag(np.linalg.inv(precision))), prior.std)
    np.testing.assert_allclose(increment_precision(SHAPE, vertical, lateral, anchor).toarray(), precision)
    with pytest.raises(ValueError, match='positive'):
        increment_precision(SHAPE, -vertical, lateral, (3., 1.5))


def test_equal_weights_give_a_plain_gaussian_increment_prior():
    cells = SHAPE[0] * SHAPE[1]
    flat = np.tile(np.linspace(0., 1., 7), (2 * cells, 1))        # members differ by a constant: no increments
    for statistic in ('mean', 'members'):
        vertical, lateral = ensemble_student_weights(flat, SHAPE, 1., SCALES, statistic)
        np.testing.assert_allclose(vertical[0], 2. / .1 ** 2)
        np.testing.assert_allclose(lateral[0], 2. / .03 ** 2)
        np.testing.assert_allclose(vertical[1], 2. / .05 ** 2)
        np.testing.assert_allclose(lateral[1], 2. / .015 ** 2)
    precision = increment_precision(SHAPE, vertical, lateral, (3., 1.5)).toarray()
    d_vertical, d_lateral = face_differences(SHAPE)
    np.testing.assert_allclose(precision[:cells, :cells], d_vertical.T @ d_vertical * 200.
                               + d_lateral.T @ d_lateral * 2. / .03 ** 2 + np.eye(cells) / 9.)


def test_ensemble_weights_at_the_mean_and_over_the_members():
    rng = np.random.default_rng(2)
    cells = SHAPE[0] * SHAPE[1]
    state = rng.normal(size=(2 * cells, 30))
    state[:cells] += np.repeat((np.arange(SHAPE[1]) >= 3) * 3., SHAPE[0]).reshape(SHAPE[1], SHAPE[0]).T.ravel()[:, None]
    for got, expected in zip(ensemble_student_weights(state, SHAPE, 1., SCALES, 'mean'),
                             student_weights(state.mean(axis=1), SHAPE, 1., SCALES)):
        np.testing.assert_allclose(got, expected)
    mean_based = ensemble_student_weights(state, SHAPE, 1., SCALES, 'mean')
    member_based = ensemble_student_weights(state, SHAPE, 1., SCALES, 'members')
    # Jensen: the member average of the squared increment is at least the squared increment of the mean.
    assert all(np.all(members <= mean + 1e-12) for members, mean in zip(member_based, mean_based))
    # The step between layers 2 and 3 of the ln Rh block is the weakest vertical coupling in both.
    assert np.all(mean_based[0][0].argmin(axis=1) == 2) and np.all(member_based[0][0].argmin(axis=1) == 2)
    fields = state.reshape(2, *SHAPE, -1)
    np.testing.assert_allclose(member_based[0], 2. / (np.array([.1, .05])[:, None, None] ** 2
                                                      + np.mean(np.diff(fields, axis=2) ** 2, axis=3)))
    with pytest.raises(ValueError, match='Unknown'):
        ensemble_student_weights(state, SHAPE, 1., SCALES, 'median')
