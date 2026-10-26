"""Data-free rolling Gaussian fields used to supply entered window cells."""

from dataclasses import dataclass
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def axis_covariance(coordinates, spacing, length, kernel):
    distance = np.abs(np.subtract.outer(coordinates, coordinates)) * spacing / length
    covariance = np.exp(-distance ** 2 if kernel == 'squared-exponential' else -distance)
    return covariance + 1e-10 * np.eye(len(coordinates))


def covariance_factor(covariance):
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.)
    if values.min() < -1e-6:
        raise ValueError('Conditional prior covariance is not positive semidefinite')
    return vectors * np.sqrt(np.maximum(values, 0.))


@dataclass
class RollingPriorBuffer:
    fields: np.ndarray
    origin: tuple
    active_shape: tuple
    padding: tuple
    spacing: tuple
    lengths: tuple
    means: tuple
    stds: tuple
    rng: np.random.Generator

    kernels = ('squared-exponential', 'exponential')

    def __post_init__(self):
        shape = tuple(size + 2 * pad for size, pad in zip(self.active_shape, self.padding))
        if (self.fields.ndim != 4 or self.fields.shape[:3] != (2, *shape)
                or not np.isfinite(self.fields).all()):
            raise ValueError('Invalid rolling prior buffer fields')

    @classmethod
    def create(cls, active_shape, spacing, lengths, means, stds, members,
               buffer_ft, seed):
        if not np.isfinite(buffer_ft) or buffer_ft <= 0 or members < 2:
            raise ValueError('Require a positive buffer and at least two members')
        padding = tuple(int(np.ceil(buffer_ft / step)) for step in spacing)
        shape = tuple(size + 2 * pad for size, pad in zip(active_shape, padding))
        rng = np.random.default_rng(seed)
        fields = np.empty((2, *shape, members))
        for field, kernel in enumerate(cls.kernels):
            factors = [np.linalg.cholesky(axis_covariance(
                np.arange(size), spacing[axis], lengths[field][axis], kernel))
                for axis, size in enumerate(shape)]
            noise = rng.standard_normal((*shape, members))
            fields[field] = means[field] + stds[field] * np.einsum(
                'ij,jkm,lk->ilm', factors[0], noise, factors[1], optimize=True)
        return cls(fields, tuple(-pad for pad in padding), tuple(active_shape),
                   padding, tuple(spacing), tuple(map(tuple, lengths)),
                   tuple(means), tuple(stds), rng)

    def _move_axis(self, axis, movement):
        if movement == 0:
            return
        size = self.fields.shape[axis + 1]
        old_coords = np.arange(size) + self.origin[axis]
        new_coords = old_coords + movement
        kept, old_index, new_index = np.intersect1d(
            old_coords, new_coords, return_indices=True)
        entered_index = np.flatnonzero(~np.isin(new_coords, kept))
        entered = new_coords[entered_index]
        result = np.empty_like(self.fields)
        for field, kernel in enumerate(self.kernels):
            source = np.moveaxis(self.fields[field], axis, 0)
            target = np.moveaxis(result[field], axis, 0)
            target[new_index] = source[old_index]
            coords = np.concatenate([kept, entered])
            cov = axis_covariance(coords, self.spacing[axis],
                                  self.lengths[field][axis], kernel)
            count = len(kept)
            conditional = cov[count:, count:].copy()
            mean = np.full((len(entered), *source.shape[1:]), self.means[field])
            if count:
                weights = cho_solve(cho_factor(cov[:count, :count], lower=True),
                                    cov[:count, count:]).T
                mean += np.einsum('ij,jkm->ikm', weights,
                                  source[old_index] - self.means[field], optimize=True)
                conditional -= weights @ cov[:count, count:]
            other = 1 - axis
            other_cov = axis_covariance(np.arange(source.shape[1]), self.spacing[other],
                                        self.lengths[field][other], kernel)
            noise = self.rng.standard_normal(mean.shape)
            target[entered_index] = mean + self.stds[field] * np.einsum(
                'ij,jkm,lk->ilm', covariance_factor(conditional), noise,
                np.linalg.cholesky(other_cov), optimize=True)
        self.fields = result
        origin = list(self.origin)
        origin[axis] += movement
        self.origin = tuple(origin)

    def advance(self, movement):
        """Shift the data-free domain, conditioning new slabs on retained prior cells."""
        for axis, amount in enumerate(movement):
            self._move_axis(axis, int(amount))

    def active_fields(self):
        px, pz = self.padding
        nx, nz = self.active_shape
        return self.fields[:, px:px + nx, pz:pz + nz].copy()

    def restart(self, posterior, movement):
        """Keep posterior overlap; supply only entered cells from the prior buffer."""
        self.advance(movement)
        result = self.active_fields()
        nx, nz = self.active_shape
        source = np.asarray(posterior).reshape(2, nx, nz, -1)
        old_x = np.arange(nx) + movement[0]
        old_z = np.arange(nz) + movement[1]
        new_x = np.flatnonzero((old_x >= 0) & (old_x < nx))
        new_z = np.flatnonzero((old_z >= 0) & (old_z < nz))
        result[:, new_x[:, None], new_z] = source[:, old_x[new_x, None], old_z[new_z]]
        return result.reshape(2 * nx * nz, -1)

    def save(self, path):
        metadata = {name: getattr(self, name) for name in
                    ('origin', 'active_shape', 'padding', 'spacing', 'lengths', 'means', 'stds')}
        metadata['rng_state'] = self.rng.bit_generator.state
        with open(path, 'xb') as stream:
            np.savez_compressed(stream, fields=self.fields, metadata=json.dumps(metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            fields = saved['fields'].copy()
        rng = np.random.default_rng()
        rng.bit_generator.state = metadata.pop('rng_state')
        metadata = {key: tuple(value) for key, value in metadata.items()}
        metadata['lengths'] = tuple(map(tuple, metadata['lengths']))
        return cls(fields=fields, rng=rng, **metadata)
