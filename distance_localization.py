"""Structured PET distance-localization inputs for the x/z cascade state."""

import pickle

import numpy as np


class FurrerBengtssonLocalization:
    def __init__(self, shape, spacing, bit_cell, ranges_ft, members, channels,
                 index, path):
        nx, nz = shape
        dx, dz = spacing
        rx, rz = ranges_ft
        if not np.isfinite(ranges_ft).all() or min(ranges_ft) <= 0:
            raise ValueError('Localization ranges must be finite and positive')
        ix, iz = bit_cell
        clearance = (min(ix, nx - 1 - ix) * dx, min(iz, nz - 1 - iz) * dz)
        if rx > clearance[0] - dx or rz > clearance[1] - dz:
            raise ValueError('Localization support must leave an untouched boundary collar')
        self.shape = shape
        self.channels = channels
        self.index = index
        # PET's first axis holds the two parameter blocks; positions are (x, z-cell, block).
        config = {'field': [2, nx, nz], 'actnum': None}
        config[('em', float(index), 'model')] = {
            'taper_func': 'fb', 'position': [[ix, iz, 0], [ix, iz, 1]],
            'range': [rz / dz, ':'], 'anisotropi': [(rx / dx) / (rz / dz), 0.]}
        with open(path, 'xb') as stream:
            pickle.dump(config, stream)
        self.input = {'field': config['field'], 'dist_loc': str(path)}
        # Imported here: PET is only needed when a localization is built.
        import pandas as pd
        from pipt.localization import DistanceLocalization
        self.prior_info = {'model': {'active': 2 * nx * nz, 'nx': nx, 'ny': 2, 'nz': nz}}
        observed = pd.DataFrame({'em': [np.empty(channels)]}, index=[float(index)])
        self.pet = DistanceLocalization(info=self.input, data=observed, parameters=['model'],
                                        ensemble_size=members, prior_info=self.prior_info)
        mask = self.pet()
        self.weights = mask[:, 0].toarray().ravel()
        if self.weights.shape != (2 * nx * nz,) or np.any(self.weights < 0):
            raise ValueError('PET mask does not match the stacked state')
        field = self.weights[:nx * nz].reshape(nx, nz)
        if (np.any(field[[0, -1]]) or np.any(field[:, [0, -1]])
                or not 0 < field[ix, iz] <= 1):
            raise ValueError('PET mask orientation or protected collar is invalid')

    def configure(self, updater):
        updater.keys_da = {'localization': self.input}
        updater.localization = self.pet
        updater.list_states = ['model']
        updater.prior_info = self.prior_info


def blend_localized_samples(incoming, proposal, weights):
    """Retain member-wise boundary values while tapering the proposed change."""
    result = incoming + weights[:, None] * (proposal - incoming)
    result[weights == 0] = incoming[weights == 0]
    return result


def localized_rtps(ensemble, reference_std, weights, strength):
    """Relax anomalies toward reference spread without shrinking or changing the mean."""
    ensemble = np.asarray(ensemble, dtype=float)
    reference_std = np.asarray(reference_std, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (ensemble.ndim != 2 or ensemble.shape[1] < 2
            or reference_std.shape != (ensemble.shape[0],)
            or weights.shape != reference_std.shape
            or not np.isfinite(ensemble).all() or not np.isfinite(reference_std).all()
            or not np.isfinite(weights).all() or np.any(reference_std < 0)
            or np.any((weights < 0) | (weights > 1))
            or not np.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('RTPS requires finite ensemble/spreads, weights and strength in [0, 1]')
    mean = ensemble.mean(axis=1, keepdims=True)
    before = ensemble.std(axis=1, ddof=1)
    target = before + strength * weights * np.maximum(reference_std - before, 0.)
    factors = np.ones_like(before)
    active = (target > before) & (before > np.finfo(float).eps)
    factors[active] = target[active] / before[active]
    result = ensemble.copy()
    result[active] = mean[active] + factors[active, None] * (ensemble[active] - mean[active])
    diagnostics = {'std_before': before, 'std_target': target,
                   'std_reference': reference_std, 'inflation_factors': factors,
                   'zero_spread_unrestored': (target > before) & ~active}
    return result, diagnostics
