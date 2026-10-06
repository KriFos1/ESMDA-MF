"""Unified moving-window domain: lateral prior, conditional entry, exit."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_DA_sequential as sequential


BACKGROUND = float(np.log(1.5))
RESERVOIR = float(np.log(30.))
MODE_STD = 0.23


def column_correlation(rh, nx, column_a, column_b):
    a = rh[column_a].ravel()
    b = rh[column_b].ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean()))


def test_sample_prior_2d_is_blocky_and_laterally_correlated(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (9, 1, 12))
    np.random.seed(7)
    ne = 240
    zero_d = np.full(ne, RESERVOIR)
    state, covariance, mean = sequential.sample_prior_2d(
        ne, zero_d, BACKGROUND, RESERVOIR, MODE_STD, .3, .05, projection=False)
    nx, nz = 9, 12
    assert state.shape == (2 * nx * nz, ne)
    assert covariance.shape == (2 * nx * nz, 2 * nx * nz)
    rh = state[:nx * nz].reshape(nx, nz, ne)
    within = ((abs(rh - BACKGROUND) < 5 * MODE_STD) | (abs(rh - RESERVOIR) < 5 * MODE_STD))
    assert within.mean() > .95
    bit = nx // 2
    adjacent = column_correlation(rh, nx, bit, bit + 1)
    far = column_correlation(rh, nx, bit, nx - 1)
    assert adjacent > .75
    assert far < adjacent - .1


def test_shift_window_state_drops_exited_and_shifts_interior(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (6, 1, 5))
    np.random.seed(11)
    nx, nz, ne = 6, 5, 3
    cells = nx * nz
    base_levels = RESERVOIR + .001 * np.arange(nx)
    rh = np.tile(base_levels[:, None, None], (1, nz, ne))
    latent = rh + .5
    state = np.vstack([rh.reshape(cells, ne), latent.reshape(cells, ne)])
    shifted = sequential.shift_window_state(state, (2, 0), BACKGROUND, RESERVOIR,
                                            MODE_STD, .25, .05)
    new_rh = shifted[:cells].reshape(nx, nz, ne)
    new_latent = shifted[cells:].reshape(nx, nz, ne)
    for column in range(nx - 2):
        np.testing.assert_allclose(new_rh[column], rh[column + 2])
        np.testing.assert_allclose(new_latent[column], latent[column + 2])
    for column in (nx - 2, nx - 1):
        sample = new_rh[column].ravel()
        near_modes = ((abs(sample - BACKGROUND) < 6 * MODE_STD)
                      | (abs(sample - RESERVOIR) < 6 * MODE_STD)).mean()
        assert near_modes > .9
        assert not np.allclose(sample, new_rh[column - 1].ravel())


def test_shift_window_state_negative_movement_and_vertical_entry(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (5, 1, 6))
    np.random.seed(13)
    nx, nz, ne = 5, 6, 2
    cells = nx * nz
    base = RESERVOIR + .001 * np.arange(nx)[:, None] + .0001 * np.arange(nz)[None, :]
    rh = np.tile(base[..., None], (1, 1, ne))
    state = np.vstack([rh.reshape(cells, ne), np.zeros((cells, ne))])
    left = sequential.shift_window_state(state, (-1, 0), BACKGROUND, RESERVOIR, MODE_STD, .25, .05)
    left_rh = left[:cells].reshape(nx, nz, ne)
    for column in range(1, nx):
        np.testing.assert_allclose(left_rh[column], rh[column - 1])
    down = sequential.shift_window_state(state, (0, 2), BACKGROUND, RESERVOIR, MODE_STD, .25, .05)
    down_rh = down[:cells].reshape(nx, nz, ne)
    for row in range(nz - 2):
        np.testing.assert_allclose(down_rh[:, row], rh[:, row + 2])
    entered = down_rh[:, nz - 2:].ravel()
    near_modes = ((abs(entered - BACKGROUND) < 6 * MODE_STD)
                  | (abs(entered - RESERVOIR) < 6 * MODE_STD)).mean()
    assert near_modes > .9


def test_shift_window_state_rejects_malformed_state(monkeypatch):
    monkeypatch.setattr(sequential, 'dims', (5, 1, 6))
    with pytest.raises(ValueError):
        sequential.shift_window_state(np.zeros((7, 2)), (1, 0), BACKGROUND, RESERVOIR,
                                      MODE_STD, .25, .05)


def test_conditional_columns_preserve_modes_and_flip_rarely():
    np.random.seed(17)
    nz, ne = 30, 500
    modes = np.where(np.arange(nz)[:, None] % 7 < 3, RESERVOIR, BACKGROUND)
    base = np.repeat(modes, ne, axis=1)
    base_latent = np.zeros((nz, ne))
    rh, latent = sequential.conditional_columns(base, base_latent, BACKGROUND, RESERVOIR,
                                                 MODE_STD, .25, .05)
    midpoint = .5 * (BACKGROUND + RESERVOIR)
    unchanged = (rh > midpoint) == (base > midpoint)
    flips = 1. - unchanged.mean()
    assert .01 < flips < .15
    assert abs(rh[unchanged & (base > midpoint)].mean() - RESERVOIR) < .05
    assert abs(rh[unchanged & (base <= midpoint)].mean() - BACKGROUND) < .05
    assert abs(latent.mean()) < .05
    assert .15 < latent.std() < .4
