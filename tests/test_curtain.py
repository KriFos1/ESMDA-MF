import colorsys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pytest

from plot_curtain import (hls_lightness_rgb, member_fields, paint_point,
                         posterior_moments, uncertainty_rgba)


def test_hls_variance_mapping_preserves_hue_and_saturation_and_lightens():
    rgb = (0.12, 0.38, 0.78)
    unchanged = hls_lightness_rgb(rgb, 0.0, strength=0.8)
    fogged = hls_lightness_rgb(rgb, 0.25, strength=0.8)
    at_threshold = hls_lightness_rgb(rgb, 0.35, strength=0.8)
    clipped = hls_lightness_rgb(rgb, 0.350001, strength=0.8)
    hue0, lightness0, saturation0 = colorsys.rgb_to_hls(*rgb)
    hue1, lightness1, saturation1 = colorsys.rgb_to_hls(*fogged)

    np.testing.assert_allclose(unchanged, rgb)
    assert lightness1 > lightness0
    assert hue1 == pytest.approx(hue0)
    assert saturation1 == pytest.approx(saturation0)
    assert at_threshold != (1., 1., 1.)
    assert clipped == (1., 1., 1.)


def test_uncertainty_rgba_keeps_missing_cells_transparent():
    field = np.array([[0.2, 0.8], [0.5, np.nan]])
    variance = np.array([[0.0, 0.5], [1.0, np.nan]])
    rgba = uncertainty_rgba(field, variance, plt.get_cmap("viridis"),
                            Normalize(0.0, 1.0), variance_scale=1.0, strength=.8)

    expected = plt.get_cmap("viridis")(Normalize(0.0, 1.0)(field[0, 0]))
    np.testing.assert_allclose(rgba[0, 0], expected)
    assert rgba[1, 1, 3] == 0.
    np.testing.assert_array_equal(rgba[1, 0, :3], (1., 1., 1.))
    np.testing.assert_array_equal(rgba[0, 1, :3], (1., 1., 1.))


def test_posterior_moments_use_log10_rh_and_transformed_rv_variance():
    dims = (2, 1, 2)
    members = np.array([
        [0.1, 0.2, 0.3, 0.4, -0.5, 0.0, 0.5, 1.0],
        [0.2, 0.1, 0.4, 0.3, -0.4, 0.1, 0.6, 0.9],
        [0.0, 0.3, 0.2, 0.5, -0.6, -0.1, 0.4, 1.1],
    ])
    rh, rv, variance_rh, variance_rv, is_2d = posterior_moments(
        members, dims, ratio_bounds=(1., 3.), rv_cap_ohmm=80.)
    expected = [member_fields(member, dims, (1., 3.), 80.) for member in members]
    expected_rh = np.stack([entry[0] / np.log(10.) for entry in expected])
    expected_rv = np.stack([entry[1] / np.log(10.) for entry in expected])
    np.testing.assert_allclose(rh, np.mean([entry[0] for entry in expected], axis=0))
    np.testing.assert_allclose(rv, np.mean([entry[1] for entry in expected], axis=0))
    np.testing.assert_allclose(variance_rh, np.var(expected_rh, axis=0, ddof=1))
    np.testing.assert_allclose(variance_rv, np.var(expected_rv, axis=0, ddof=1))
    assert is_2d


def test_paint_point_places_1d_uncertainty_only_at_bit_column():
    dims = (3, 1, 2)
    members = [
        np.array([0.1, 0.2, -0.5, 0.5]),
        np.array([0.2, 0.3, -0.4, 0.6]),
        np.array([0.0, 0.4, -0.6, 0.4]),
    ]
    result = {"dims": dims, "spacing_ft": (1., 1., 1.),
              "shift_ft": {"x": 0., "z": 0.}, "posterior_params": members,
              "rv_ratio_bounds": (1., 3.), "rv_hard_cap_ohmm": 80.}
    rh = np.full((3, 2), np.nan)
    rv = np.full((3, 2), np.nan)
    variance_rh = np.full((3, 2), np.nan)
    variance_rv = np.full((3, 2), np.nan)
    coverage = np.zeros((3, 2))

    paint_point(rh, rv, coverage, result, (0., 0.), (1., 1.),
                variance_rh, variance_rv)

    assert np.isnan(rh[[0, 2]]).all()
    assert np.isfinite(rh[1]).all()
    assert np.isnan(variance_rh[[0, 2]]).all()
    assert np.isfinite(variance_rh[1]).all()
