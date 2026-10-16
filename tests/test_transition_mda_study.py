import numpy as np

from transition_mda_study import inflation_schedule, rh_rmse


def test_ten_step_inflation_has_unit_total_likelihood_weight():
    inflation = inflation_schedule()
    assert inflation.shape == (10,)
    assert np.allclose(np.diff(inflation), np.diff(inflation)[0])
    assert np.all(np.diff(inflation) < 0)
    assert sum(1 / inflation[:3]) < 1
    np.testing.assert_allclose(sum(1 / inflation), 1.)


def test_truth_rmse_preserves_x_z_orientation_and_log_units():
    x = np.array([.5, 1.5, 2.5])
    z = np.array([.5, 1.5])
    rh_z_x = np.array([[1., 4., 16.], [2., 8., 32.]])
    saved = {'dims': (3, 1, 2), 'spacing_ft': (1., 1., 1.),
             'shift_ft': {'x': 0., 'z': 0.}}
    exact = np.log(rh_z_x.T).reshape(6, 1)
    np.testing.assert_allclose(rh_rmse(exact, saved, (x, z, rh_z_x, {})), 0., atol=1e-14)
    shifted = exact + np.log(10.)
    np.testing.assert_allclose(rh_rmse(shifted, saved, (x, z, rh_z_x, {})), 1.)
