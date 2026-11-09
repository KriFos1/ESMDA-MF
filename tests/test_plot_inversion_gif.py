import pickle

import numpy as np
import pytest

import plot_inversion_gif as gif


def write_chain(run_dir, points=3, nx=3, nz=4):
    rng = np.random.default_rng(5)
    for index in range(points):
        ensemble = rng.normal(size=(6, 2 * nx * nz))
        result = {'dims': (nx, 1, nz), 'spacing_ft': (5., 5., 5.), 'shift_ft': {'x': 100. * index, 'z': 0.},
                  'posterior_ensemble_params': ensemble, 'posterior_params': ensemble.copy(),
                  'rv_ratio_bounds': (1., 3.), 'anisotropy': 'log_clipped', 'selected_fidelity': '2D'}
        with open(run_dir / f'inversion_results_assim_{index}.pkl', 'wb') as stream:
            pickle.dump(result, stream)


def write_trajectory(path, points=3):
    rows = [[point + 1, 8000. + point, 10000. + 100. * point, 85., 90., 100. * point, 0.] for point in range(points)]
    path.write_text('% header\n' + '\n'.join(' '.join(f'{v:.4f}' for v in row) for row in rows) + '\n')


def test_field_gif_without_benchmark_labels_the_axis_by_measured_depth(tmp_path, monkeypatch):
    write_chain(tmp_path)
    trajectory = tmp_path / 'trajectory.DAT'
    write_trajectory(trajectory)
    with pytest.raises(SystemExit):
        gif.parse_args(['--run-dir', str(tmp_path), '--no-benchmark', '--x-axis', 'md'])
    lwd = tmp_path / 'LWD_resistivity.txt'
    lwd.write_text('% logpt Res1 Res2\n1 2.0 4.0\n2 20.0 40.0\n3 200.0 400.0\n')
    with pytest.raises(SystemExit):
        gif.parse_args(['--run-dir', str(tmp_path), '--no-benchmark', '--lwd', str(lwd)])
    args = gif.parse_args(['--run-dir', str(tmp_path), '--no-benchmark', '--x-axis', 'md',
                           '--trajectory', str(trajectory), '--lwd', str(lwd), '--dpi', '25',
                           '--output', str(tmp_path / 'out.gif')])
    monkeypatch.setattr(gif, 'load_benchmark', lambda *a: pytest.fail('loaded a benchmark'))
    titles, red_dots = [], []
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    savefig = plt.Figure.savefig
    def inspect(fig, *a, **k):
        titles.append((fig._suptitle.get_text(), [axis.get_xlabel() for axis in fig.axes]))
        dots = {}
        for name, profile in zip(('Rh', 'Rv'), fig.axes[-2:]):          # the Rh and Rv bit-column panels
            red = [c for c in profile.collections if hasattr(c, 'get_offsets') and len(c.get_offsets())
                   and tuple(c.get_facecolor()[0]) == to_rgba('red')]
            dots[name] = [tuple(map(float, offset)) for c in red for offset in c.get_offsets()]
            assert not [line for line in profile.get_lines() if line.get_color() == 'red']       # no red line
        red_dots.append(dots)
        return savefig(fig, *a, **k)
    monkeypatch.setattr(plt.Figure, 'savefig', inspect)
    output, count = gif.render(args)
    assert output.is_file() and count == 3
    from PIL import Image
    with Image.open(output) as image:
        assert image.n_frames == 3
    # The bit of point 1 sits at plane x = 107.5 ft (window centre); its MD label follows the trajectory.
    assert 'MD = 10108 ft' in titles[1][0] and 'MD [ft]' in titles[1][1]
    # Frame 1: one red dot per panel at the bit TVD, Res1 (LWD Rh) in the Rh panel, Res2 (LWD Rv) in the Rv panel.
    assert red_dots[1]['Rh'] == [(pytest.approx(np.log10(20.)), 8001.)]
    assert red_dots[1]['Rv'] == [(pytest.approx(np.log10(40.)), 8001.)]


def test_benchmark_gif_still_requires_no_trajectory(tmp_path):
    args = gif.parse_args(['--run-dir', str(tmp_path)])
    assert args.x_axis == 'plane' and args.trajectory is None and args.benchmark is not None
