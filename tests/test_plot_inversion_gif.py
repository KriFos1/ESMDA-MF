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
    assert args.view_points is None and args.view_bit_position is None and not args.plain_panels


def test_view_limits_keep_the_bit_in_place_and_extrapolate_beyond_the_well():
    positions = 10. * np.arange(50)                                     # logging points 10 ft apart
    assert gif.view_limits(positions, 30, 20, 15) == (150., 350.)       # 15 points behind, 5 ahead
    assert gif.view_limits(positions, 3, 20, 15) == (-120., 80.)        # the view starts before the first point
    assert gif.view_limits(positions, 48, 20, 15) == (330., 530.)       # and ends past the last one
    uneven = np.array([0., 10., 25., 45.])
    assert gif.view_limits(uneven, 2, 4, 3) == (-10., 45.)              # end spacings: 10 ft before, 20 ft after
    assert gif.view_limits(uneven, 3, 4, 3) == (0., 65.)
    edges = np.arange(0., 101., 5.)                                     # 20 canvas columns of 5 ft
    assert gif.view_columns(edges, (12., 31.)) == (2, 7)
    assert gif.view_columns(edges, (-50., 1000.)) == (0, 20)
    first, last = gif.view_columns(edges, (200., 300.))                 # outside the canvas: nothing to draw
    assert last <= first


def test_view_options_are_validated(tmp_path):
    base = ['--run-dir', str(tmp_path), '--no-benchmark']
    assert gif.parse_args([*base, '--view-points', '100']).view_bit_position == 75
    assert gif.parse_args([*base, '--view-points', '100', '--view-bit-position', '60']).view_bit_position == 60
    for bad in (['--view-bit-position', '3'], ['--view-points', '1'],
                ['--view-points', '10', '--view-bit-position', '11']):
        with pytest.raises(SystemExit):
            gif.parse_args([*base, *bad])


@pytest.mark.parametrize('with_trajectory', [True, False])
def test_moving_view_follows_the_bit_and_plain_panels_drop_the_variance_wash(tmp_path, monkeypatch, with_trajectory):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    write_chain(tmp_path, points=6)
    options = ['--run-dir', str(tmp_path), '--no-benchmark', '--view-points', '4', '--view-bit-position', '3',
               '--plain-panels', '--dpi', '25', '--output', str(tmp_path / 'out.gif')]
    if with_trajectory:
        trajectory = tmp_path / 'trajectory.DAT'
        write_trajectory(trajectory, points=6)
        options += ['--x-axis', 'md', '--trajectory', str(trajectory)]
        positions = 100. * np.arange(6)                 # the trajectory's logging points
    else:
        positions = 100. * np.arange(6) + 7.5           # the checkpoints' window centres
    args = gif.parse_args(options)
    frames = []
    savefig = plt.Figure.savefig
    def inspect(fig, *a, **k):
        maps = fig.axes[:4]
        frames.append({'xlim': [axis.get_xlim() for axis in maps], 'titles': [axis.get_title() for axis in maps],
                       'xlabels': [axis.get_xlabel() for axis in maps], 'ylabels': [axis.get_ylabel() for axis in maps],
                       'faces': [to_rgba(axis.get_facecolor()) for axis in maps],
                       'images': [np.asarray(axis.images[0].get_array()) if axis.images else None for axis in maps],
                       'axes': len(fig.axes), 'profiles': [axis.get_xlabel() for axis in fig.axes[-2:]],
                       'suptitle': fig._suptitle.get_text(),
                       'legend': [text.get_text() for text in fig.legends[0].get_texts()]})
        return savefig(fig, *a, **k)
    monkeypatch.setattr(plt.Figure, 'savefig', inspect)
    output, count = gif.render(args)
    assert output.is_file() and count == 6 and len(frames) == 6
    for index, frame in enumerate(frames):
        expected = gif.view_limits(positions, index, 4, 3)
        assert all(limits == pytest.approx(expected) for limits in frame['xlim'])      # all four maps share the view
        assert (positions[index] - expected[0]) / (expected[1] - expected[0]) == pytest.approx(.75)
    last = frames[-1]
    assert last['xlim'][0] == pytest.approx((positions[2], 2 * positions[5] - positions[4]))  # one point past the end
    assert last['titles'] == ['Estimated log10 Rh (at bit), variance wash', 'Estimated log10 Rv (at bit), variance wash',
                              'Estimated log10 Rh (at bit), no variance wash',
                              'Estimated log10 Rv (at bit), no variance wash']
    position_label = 'MD [ft]' if with_trajectory else 'X [ft]'
    assert last['xlabels'] == ['', '', position_label, position_label]
    assert last['ylabels'] == ['TVD [ft]', '', 'TVD [ft]', '']
    assert last['axes'] == 8 and last['profiles'] == ['log10 Rh [ohm m]', 'log10 Rv [ohm m]']      # 4 maps, 2 bars, 2 columns
    assert all(face == to_rgba(gif.UNPAINTED) for face in last['faces'])
    assert 'view of 4 logging points: 3 behind the bit, 1 ahead' in last['suptitle']
    assert 'not yet inverted' in last['legend']
    white = np.array([1., 1., 1., 1.])
    for washed, plain in zip(last['images'][:2], last['images'][2:]):
        painted = washed[..., 3] > 0
        assert painted.any() and np.array_equal(painted, plain[..., 3] > 0)        # same cells painted
        whitened = np.all(washed == white, axis=-1)
        assert whitened.any()                                                      # the wash blanks uncertain cells
        assert not np.all(plain == white, axis=-1).any()                           # the plain map shows them
        assert np.array_equal(washed[painted & ~whitened][:, 3], plain[painted & ~whitened][:, 3])


def test_gif_skips_a_newest_checkpoint_that_is_still_being_written(tmp_path, capsys):
    write_chain(tmp_path, points=3)
    complete = (tmp_path / 'inversion_results_assim_2.pkl').read_bytes()
    (tmp_path / 'inversion_results_assim_3.pkl').write_bytes(complete[:len(complete) // 3])
    args = gif.parse_args(['--run-dir', str(tmp_path), '--no-benchmark', '--dpi', '25',
                           '--output', str(tmp_path / 'out.gif')])
    output, count = gif.render(args)
    assert count == 3 and 'still being written' in capsys.readouterr().out
