import numpy as np

import run_DA_sequential as sequential
from plot_curtain import member_fields


def test_curtain_rv_matches_the_2d_forward_transform(monkeypatch):
    dims = (2, 5, 3)
    monkeypatch.setattr(sequential, 'dims', dims)
    monkeypatch.setattr(sequential, 'min_ratio', 1.)
    monkeypatch.setattr(sequential, 'max_ratio', 3.)
    monkeypatch.setattr(sequential, 'log_rv_cap', np.log(40.))
    parameters = np.r_[np.log([2., 5., 20., 30., 35., 40.]),
                       np.array([-2., -.5, 0., .5, 2., 4.])]
    field, _, _ = sequential._build_param_dict(parameters)
    plotted_rh, plotted_rv, _ = member_fields(parameters, dims, (1., 3.), 40.)
    simulated_rh = field['rh'].reshape(dims, order='F')[:, 0, :]
    simulated_rv = field['rv'].reshape(dims, order='F')[:, 0, :]
    np.testing.assert_allclose(plotted_rh, simulated_rh)
    np.testing.assert_allclose(plotted_rv, simulated_rv)
