import json

import h5py
import numpy as np
import pandas as pd
import pytest

import run_DA_sequential as sequential
import sequential_matern_1d as driver


@pytest.fixture
def gate_environment(tmp_path, monkeypatch):
    """Fake 1D chain where index 2 asks for ln Rh = 10, beyond the clamp: the probe must fail there."""
    benchmark = tmp_path / 'benchmark'
    benchmark.mkdir()
    depths_m = np.array([100., 101.524, 103.048, 104.572])
    nx, nz = 5, 6
    with h5py.File(benchmark / 'globalmodel.h5', 'w') as handle:
        handle['wellpath/Z'] = depths_m[:, None]
        handle['wellpath/X'] = (np.arange(4) * 1.524)[:, None]
        handle['wellpath/Y'] = np.zeros((4, 1))
        handle['formation/x'] = (np.arange(-10, 30) * 1.524)[:, None]
        handle['formation/z'] = (np.arange(60, 140) * 1.524)[:, None]
        handle['formation/Rh'] = np.full((80, 3, 40), 1.5)
    tool = ('6kHz', '83ft')
    depths = depths_m * sequential.meter_to_feet
    rows = [np.full(8, 2.), np.full(8, 2.), np.full(8, 10.), np.full(8, 2.)]
    pd.DataFrame({tool: rows}, index=depths).to_pickle(tmp_path / 'data.pkl')
    pd.DataFrame({tool: [['ABS', [.25] * 8]] * 4}, index=depths).to_pickle(tmp_path / 'var.pkl')
    manifest = {'benchmark': str(benchmark), 'dims': [nx, 1, nz], 'spacing_ft': [5., 5., 5.],
                'data': str(tmp_path / 'data.pkl'), 'variance': str(tmp_path / 'var.pkl'),
                'prior_background': np.log10(1.5), 'prior_reservoir': np.log10(30.),
                'prior_mode_std': .1, 'prior_reservoir_fraction': .15, 'prior_ratio_max': 3.,
                'prior_latent_correlation_x_ft': 80., 'prior_latent_correlation_z_ft': 35.,
                'dx_ft': 5., 'dz_ft': 5., 'rv_parameterization': {'latent_std': .12}}
    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'run.json').write_text(json.dumps(manifest))
    well = {'X': np.arange(4)[:, None] * 1.524, 'Z': depths_m[:, None]}

    def setup(benchmark_dir, tools):
        monkeypatch.setattr(sequential, 'TVD', depths, raising=False)
        monkeypatch.setattr(sequential, 'MD', depths, raising=False)
        monkeypatch.setattr(sequential, 'WX', np.arange(4) * 5., raising=False)
        monkeypatch.setattr(sequential, 'WY', np.zeros(4), raising=False)
        monkeypatch.setattr(sequential, 'wellpath', well, raising=False)

    def fake_pred(param, index, simfidelity='1D'):
        return np.full(8, float(np.mean(param[:nz])))

    monkeypatch.setattr(sequential, 'setup_simulators', setup)
    monkeypatch.setattr(sequential, 'simulate_pred_only', fake_pred)
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    for name in ('dims', 'dX', 'dY', 'dZ', 'tools', 'max_ratio', 'log_rv_cap', 'anisotropy',
                 'current_shift'):   # the driver sets these; restore them for later tests
        monkeypatch.setattr(sequential, name, getattr(sequential, name, None), raising=False)
    return tmp_path, reference
