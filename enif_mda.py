"""EnIF-MDA steps for one logging point, through PET's graph-informed information filter.

PET's ``enif`` analysis (``pipt.update_schemes.analysis.enif``, PET >= 0.3)
replaces the ensemble covariance of each ES-MDA step by a sparse precision
and a boosted linear regression of the responses on the standardised state.
At the first MDA step it fits the precision on a parameter graph; later steps
carry the posterior precision forward (information-filter recursion).

Here the prior precision is known, the Matern (SPDE) precision of the chain's
prior, so it is fed directly: the scheme's preceding-information slot is
seeded at step 0 with that precision in physical units (scales of one), and
PET skips its precision fit. Rows that lose all spread (members clamped to the
same bound) are held fixed by PET; the carried precision is then conditioned
on them, i.e. restricted to the remaining rows.
"""

from types import SimpleNamespace

import numpy as np
from scipy import sparse


def active_rows(state):
    """PET's active rows: finite in every member and with some ensemble spread."""
    active = np.all(np.isfinite(state), axis=1)
    active[active] = np.ptp(state[active], axis=1) > 0
    return np.flatnonzero(active)


class EnIFProbe:
    """PET's EnIF analysis for one point's MDA, seeded with a known prior precision.

    Parameters
    ----------
    precision : sparse matrix, (n, n)
        Prior precision of the state in physical units.
    observations, variance : arrays, (nd,)
        Observations and their (un-inflated) variances.
    inflation : sequence of float
        The MDA inflation factors of this point, one per step.
    seed : int
        Seed of PET's extra-noise generator (unexplained regression variance).
    """

    def __init__(self, precision, observations, variance, inflation, seed):
        from pipt.update_schemes.analysis.enif import _EnIFInformation, enif_update

        self._information = _EnIFInformation
        self.precision = sparse.csc_array(precision)
        n = self.precision.shape[0]
        self.groups = (('state', (0, n)),)
        self.scheme = SimpleNamespace(
            keys_da={'enif': {}}, keys_en={'disable_tqdm': True}, idX={'state': (0, n)},
            prior_info={'state': {}}, alpha=[float(a) for a in inflation], iteration=0,
            vecObs=np.asarray(observations, dtype=float), cov_data=np.asarray(variance, dtype=float),
            ensemble=SimpleNamespace(rng=np.random.RandomState(seed)), enif_information=None)
        self.analysis = enif_update(self.scheme)

    def _seed(self, rows):
        self.scheme.enif_information = self._information(
            precision=self.precision[rows][:, rows].tocsc(), scales=np.ones(len(rows)), active_rows=rows,
            groups=self.groups, iteration=self.scheme.iteration)

    def _condition(self, rows):
        """Restrict the carried precision to rows that kept their spread."""
        information = self.scheme.enif_information
        keep = np.isin(information.active_rows, rows)
        if not np.array_equal(information.active_rows[keep], rows):
            raise ValueError('EnIF rows regained ensemble spread after being held fixed')
        index = np.flatnonzero(keep)
        inverse = sparse.diags_array(1. / information.scales[index], format='csc')
        carried = sparse.csr_array(information.precision)[index][:, index]
        self.scheme.enif_information = self._information(
            precision=(inverse @ carried @ inverse).tocsc(), scales=np.ones(len(rows)), active_rows=rows,
            groups=self.groups, iteration=self.scheme.iteration)

    def step(self, state, predicted, perturbed):
        """One MDA step: the additive state step for ``state`` (n, ne)."""
        rows = active_rows(state)
        information = self.scheme.enif_information
        if information is None:
            self._seed(rows)
        elif not np.array_equal(information.active_rows, rows):
            self._condition(rows)
        step = self.analysis.update(state, predicted, perturbed).step
        self.scheme.iteration += 1
        return step
