"""Bridge to PET's update API (PET >= 0.3, Python >= 3.12).

PET now builds an analysis against a scheme (``approx_update(scheme)``) and returns an
``AnalysisResult``; this keeps the scheme-as-attribute-holder interface the study code uses.
"""

from types import SimpleNamespace


class ApproxUpdater:
    """Holds the scheme attributes ``approx_update`` reads; ``update`` stores the step in ``step``."""

    def __init__(self):
        self.lam = 0.
        self.trunc_energy = .95
        self.localization = SimpleNamespace(name='none')
        self.keys_da = {}
        self.step = None

    @property
    def keys_da(self):
        return self._keys_da

    @keys_da.setter
    def keys_da(self, value):
        # Without a localization entry the update is unlocalized, as in earlier PET versions
        self._keys_da = value
        if 'localization' not in value:
            self.localization = SimpleNamespace(name='none')

    def update(self, enX, enY, enE, **kwargs):
        from pipt.update_schemes.analysis.approx import approx_update

        self.step = approx_update(self).update(enX, enY, enE, **kwargs).step


def auto_adaptive_localization(threshold, cells):
    """Sigmoid taper whose midpoint is the correlation ``threshold`` itself (not a noise multiple).

    ``ensemble`` projection correlates the state with the ne x ne weighted innovations, as earlier PET did.
    """
    from pipt.localization import AutoAdaptiveLocalization

    return AutoAdaptiveLocalization({'field': [int(cells), 1, 1], 'threshold': 'fixed',
                                     'cutoff': float(threshold), 'type': 'sigm',
                                     'projection': 'ensemble'})
