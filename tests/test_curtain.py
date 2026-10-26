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


def test_paint_point_prefers_transported_full_ensemble():
    result = {"dims": (1, 1, 1), "spacing_ft": (1., 1., 1.),
              "shift_ft": {"x": 0., "z": 0.},
              "posterior_params": [np.array([2., 0.])],
              "posterior_ensemble_params": [np.array([-1., 0.]), np.array([1., 0.])],
              "rv_ratio_bounds": (1., 3.), "rv_hard_cap_ohmm": 80.}
    rh = np.full((1, 1), np.nan)
    rv = np.full((1, 1), np.nan)
    coverage = np.zeros((1, 1))

    paint_point(rh, rv, coverage, result, (0., 0.), (1., 1.))

    assert rh[0, 0] == pytest.approx(0.)
    assert coverage[0, 0] == 1.


def test_nearest_bit_painting_keeps_each_cell_from_its_closest_window():
    from plot_curtain import build_curtain

    def window(shift_x, value, spread):
        members = [np.r_[np.full(3, value), np.zeros(3)], np.r_[np.full(3, value + spread), np.zeros(3)]]
        return {"dims": (3, 1, 1), "spacing_ft": (1., 1., 1.), "shift_ft": {"x": shift_x, "z": 0.},
                "posterior_ensemble_params": members, "posterior_params": members[:1],
                "rv_ratio_bounds": (1., 3.), "rv_hard_cap_ohmm": 80.}
    # Three windows moving one column per point; the bit is the middle column (world k + 1).
    checkpoints = [(k, window(float(k), float(k), .2 * (k + 1))) for k in range(3)]
    last = build_curtain(checkpoints, include_variance=True)
    nearest = build_curtain(checkpoints, include_variance=True, paint='nearest-bit')
    mean = lambda k: k + .1 * (k + 1)
    np.testing.assert_allclose(last[0][:, 0], [mean(0), mean(1), mean(2), mean(2), mean(2)])
    np.testing.assert_allclose(nearest[0][:, 0], [mean(0), mean(0), mean(1), mean(2), mean(2)])
    variance = lambda k: np.var([0., .2 * (k + 1) / np.log(10.)], ddof=1)
    np.testing.assert_allclose(nearest[5][:, 0], [variance(0), variance(0), variance(1), variance(2), variance(2)])
    np.testing.assert_allclose(last[5][:, 0], [variance(0), variance(1), variance(2), variance(2), variance(2)])
    np.testing.assert_array_equal(nearest[2], last[2])      # coverage counts every window either way
    with pytest.raises(ValueError):
        build_curtain(checkpoints, paint='first')


def test_hue_colormap_keeps_lightness_constant_so_the_wash_does_not_shift_the_value():
    from plot_curtain import hue_colormap, value_colormap
    cmap = hue_colormap(64)
    colors = cmap(np.linspace(0., 1., 64))[:, :3]
    hls = np.array([colorsys.rgb_to_hls(*color) for color in colors])
    np.testing.assert_allclose(hls[:, 1], .5, atol=1e-12)          # constant HLS lightness
    np.testing.assert_allclose(hls[:, 2], 1., atol=1e-12)          # full saturation
    assert hls[0, 0] == pytest.approx(240. / 360.) and hls[-1, 0] == pytest.approx(0.)
    assert np.all(np.diff(hls[:-1, 0]) < 0)                        # blue (low) to red (high)
    norm = Normalize(vmin=0., vmax=1.)
    values = np.linspace(0., 1., 64)
    for value in (.1, .5, .9):
        washed = hls_lightness_rgb(cmap(norm(value))[:3], .3, strength=.75, white_threshold=.5)
        hue = colorsys.rgb_to_hls(*washed)[0]
        nearest = values[np.argmin(np.abs(hls[:, 0] - hue))]       # read the value back from the hue
        assert nearest == pytest.approx(value, abs=1.5 / 63)
    assert value_colormap('viridis').name == 'viridis' and value_colormap('hue').name == 'hls_hue'


def test_trimmed_and_cached_checkpoints_render_the_same_curtain(tmp_path, monkeypatch):
    import pickle
    import plot_curtain
    rng = np.random.default_rng(4)
    nx, nz = 5, 4
    for index in range(3):
        ensemble = rng.normal(size=(6, 2 * nx * nz))
        result = {'dims': (nx, 1, nz), 'spacing_ft': (5., 5., 5.), 'shift_ft': {'x': 5. * index, 'z': 0.},
                  'posterior_ensemble_params': ensemble, 'posterior_params': ensemble.copy(),
                  'prior_params': ensemble.T.copy(), 'post_mda_params': ensemble.T.copy(),
                  'rv_ratio_bounds': (1., 3.), 'anisotropy': 'log_clipped', 'selected_fidelity': '1D'}
        with open(tmp_path / f'inversion_results_assim_{index}.pkl', 'wb') as stream:
            pickle.dump(result, stream)
    full = []
    for index in range(3):
        with open(tmp_path / f'inversion_results_assim_{index}.pkl', 'rb') as stream:
            full.append((index, pickle.load(stream)))
    cache = {}
    trimmed = plot_curtain.load_checkpoints(tmp_path, cache)
    assert all('prior_params' not in result and 'post_mda_params' not in result for _, result in trimmed)
    assert all(result['posterior_params'] is result['posterior_ensemble_params'] for _, result in trimmed)
    for paint in ('last', 'nearest-bit'):
        expected = plot_curtain.build_curtain(full, include_variance=True, paint=paint)
        actual = plot_curtain.build_curtain(trimmed, include_variance=True, paint=paint)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    loads = []
    real_load = pickle.load
    monkeypatch.setattr(plot_curtain.pickle, 'load', lambda stream: loads.append(1) or real_load(stream))
    assert [index for index, _ in plot_curtain.load_checkpoints(tmp_path, cache)] == [0, 1, 2]
    assert loads == []                                     # unchanged files come from the cache
