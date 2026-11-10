import pickle

import numpy as np
import pytest

import plot_inversion_gif as gif


def write_chain(run_dir, points=3, nx=3, nz=4, map_seconds=None):
    rng = np.random.default_rng(5)
    for index in range(points):
        ensemble = rng.normal(size=(6, 2 * nx * nz))
        result = {'dims': (nx, 1, nz), 'spacing_ft': (5., 5., 5.), 'shift_ft': {'x': 100. * index, 'z': 0.},
                  'posterior_ensemble_params': ensemble, 'posterior_params': ensemble.copy(),
                  'rv_ratio_bounds': (1., 3.), 'anisotropy': 'log_clipped', 'selected_fidelity': '2D'}
        if map_seconds is not None and map_seconds[index] is not None:
            result['laplace'] = {'map_seconds': map_seconds[index]}      # the 2D MAP solve time of the point
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
    titles, red_dots, bits, texts = [], [], [], []
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    savefig = plt.Figure.savefig
    def inspect(fig, *a, **k):
        titles.append((fig._suptitle.get_text(), [axis.get_xlabel() for axis in fig.axes]))
        texts.extend(text.get_text() for text in fig.texts)
        markers = [tuple(map(float, c.get_offsets()[0])) for axis in fig.axes[:2] for c in axis.collections
                   if c.get_zorder() == 6]                                  # the bit crosses of the Rh and Rv maps
        paths = [np.asarray(c.get_offsets()) for axis in fig.axes[:2] for c in axis.collections if c.get_zorder() == 4]
        lines = [float(line.get_ydata()[0]) for axis in fig.axes[-2:] for line in axis.get_lines()
                 if line.get_label() == 'bit TVD']
        bits.append({'markers': markers, 'paths': paths, 'tvd_lines': lines})
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
    # Point 1: the title, the bit crosses, the path and the bit TVD lines are the trajectory's logging point
    # (plane x 100 ft, TVD 8001 ft, MD 10100 ft), not the window centre (107.5 ft, 10 ft).
    assert 'MD = 10100 ft' in titles[1][0] and 'MD [ft]' in titles[1][1]
    assert bits[1]['markers'] == [pytest.approx((100., 8001.))] * 2 and bits[1]['tvd_lines'] == [8001.] * 2
    assert len(bits[1]['paths']) == 2 and all(np.allclose(path, [[0., 8000.], [100., 8001.]]) for path in bits[1]['paths'])
    # Frame 1: one red dot per panel at the bit TVD, Res1 (LWD Rh) in the Rh panel, Res2 (LWD Rv) in the Rv panel.
    assert red_dots[1]['Rh'] == [(pytest.approx(np.log10(20.)), 8001.)]
    assert red_dots[1]['Rv'] == [(pytest.approx(np.log10(40.)), 8001.)]
    assert texts and not [text for text in texts if 'inversion speed' in text]       # only with --inversion-speed


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


def is_round(values):
    """Equally spaced multiples of a 1 / 2 / 2.5 / 5 x 10^k step."""
    steps = np.diff(values)
    mantissa = steps[0] / 10. ** np.floor(np.log10(steps[0]))
    return bool(np.allclose(steps, steps[0]) and min(abs(mantissa - nice) for nice in (1., 2., 2.5, 5.)) < 1e-9
                and abs(values[0] / steps[0] - round(values[0] / steps[0])) < 1e-9)


def test_unit_locator_puts_round_metre_ticks_on_a_feet_axis():
    from field_trajectory import FEET_PER_METER
    locator = gif.UnitLocator(FEET_PER_METER)
    ticks = locator.tick_values(8400., 8000.)                           # an inverted TVD axis, in feet
    assert np.array_equal(ticks, locator.tick_values(8000., 8400.))
    inside = ticks[(ticks >= 8000.) & (ticks <= 8400.)] / FEET_PER_METER
    assert len(inside) >= 4 and is_round(inside)                        # e.g. 2440, 2460, ... m
    assert not is_round(ticks[(ticks >= 8000.) & (ticks <= 8400.)])     # and therefore not round in feet


@pytest.mark.parametrize('x_axis', ['md', 'plane'])
def test_length_unit_m_labels_the_horizontal_and_tvd_axes_in_metres(tmp_path, monkeypatch, x_axis):
    import re
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from field_trajectory import FEET_PER_METER

    write_chain(tmp_path, points=6, nz=40)                              # 200 ft of TVD
    options = ['--run-dir', str(tmp_path), '--no-benchmark', '--length-unit', 'm', '--dpi', '25',
               '--output', str(tmp_path / 'out.gif')]
    if x_axis == 'md':                                                  # moving view, 2 x 2 maps, MD axis
        trajectory = tmp_path / 'trajectory.DAT'
        write_trajectory(trajectory, points=6)
        options += ['--x-axis', 'md', '--trajectory', str(trajectory), '--view-points', '4',
                    '--view-bit-position', '3', '--plain-panels']
        maps, with_x_ticks, with_y_ticks = 4, [2, 3], [0, 2]
        xlabels, ylabels, position = ['', '', 'MD [m]', 'MD [m]'], ['TVD [m]', '', 'TVD [m]', ''], 'MD = 3078.5 m'
        plane_of = lambda length: length * FEET_PER_METER - 10000.      # MD [m] -> plane x [ft]
    else:                                                               # whole curtain, stacked maps, plane axis
        maps, with_x_ticks, with_y_ticks = 2, [0, 1], [0, 1]
        xlabels, ylabels, position = ['', 'X [m]'], ['TVD [m]', 'TVD [m]'], 'X = 32.8 m'
        plane_of = lambda length: length * FEET_PER_METER
    args = gif.parse_args(options)
    assert args.length_unit == 'm' and gif.parse_args(options[:3]).length_unit == 'ft'
    with pytest.raises(SystemExit):
        gif.parse_args([*options, '--length-unit', 'km'])
    frames = []
    savefig = plt.Figure.savefig
    def ticks(axis):
        return [(tick.get_loc(), tick.label1.get_text()) for tick in axis.get_major_ticks()]
    def inspect(fig, *a, **k):
        fig.canvas.draw()
        map_axes, profiles = fig.axes[:maps], fig.axes[-2:]
        frames.append({'texts': [text.get_text() for text in fig.findobj(Text)],
                       'xlabels': [axis.get_xlabel() for axis in map_axes],
                       'ylabels': [axis.get_ylabel() for axis in map_axes],
                       'profile_ylabels': [axis.get_ylabel() for axis in profiles],
                       'profile_titles': [axis.get_title() for axis in profiles],
                       'suptitle': fig._suptitle.get_text(),
                       'x': [(sorted(map_axes[place].get_xlim()), ticks(map_axes[place].xaxis)) for place in with_x_ticks],
                       'y': [(sorted(axis.get_ylim()), ticks(axis.yaxis))
                             for axis in [*(map_axes[place] for place in with_y_ticks), profiles[0]]]})
        return savefig(fig, *a, **k)
    monkeypatch.setattr(plt.Figure, 'savefig', inspect)
    output, count = gif.render(args)
    assert output.is_file() and count == 6 and len(frames) == 6
    for frame in frames:
        assert not [text for text in frame['texts'] if re.search(r'\bft\b', text)]      # nothing is left in feet
        assert frame['xlabels'] == xlabels and frame['ylabels'] == ylabels
        assert frame['profile_ylabels'] == ['TVD [m]', '']
        for to_feet, axes in ((plane_of, frame['x']), (lambda length: length * FEET_PER_METER, frame['y'])):
            for (low, high), marks in axes:
                shown = [(location, text) for location, text in marks if low <= location <= high]
                assert len(shown) >= 2 and is_round([float(text) for _, text in shown])     # round metres ...
                for location, text in shown:
                    assert location == pytest.approx(to_feet(float(text)), abs=1e-6)        # ... at their place in feet
    # Point 1: MD 10100 ft in the trajectory; without one the window centre at plane x = 107.5 ft.
    assert position in frames[1]['suptitle']
    assert all(title.endswith(f'bit column at {position}') for title in frames[1]['profile_titles'])


def test_color_and_speed_options_are_validated(tmp_path):
    base = ['--run-dir', str(tmp_path), '--no-benchmark']
    args = gif.parse_args(base)
    assert args.color_scale == 'log10' and args.color_limits is None and not args.inversion_speed
    assert gif.parse_args([*base, '--color-scale', 'linear', '--color-limits', '1', '12']).color_limits == [1., 12.]
    assert gif.parse_args([*base, '--color-scale', 'linear', '--color-limits', '0', '12']).color_limits == [0., 12.]
    for bad in (['--color-limits', '12', '1'], ['--color-limits', '0', '12'],              # log10 needs positive limits
                ['--color-scale', 'linear', '--color-limits', '3', '3'], ['--color-scale', 'sqrt'],
                ['--inversion-speed']):                                                    # needs the trajectory's MD
        with pytest.raises(SystemExit):
            gif.parse_args([*base, *bad])


def test_inversion_speed_is_the_md_step_over_the_2d_solve_time():
    meta = [{'index': 0, 'map_seconds': 50.}, {'index': 2, 'map_seconds': None}, {'index': 3, 'map_seconds': 4.}]
    measured_depth = np.array([100., 103., 106., 110.])                 # [m]
    speeds = gif.inversion_speeds(meta, measured_depth)
    assert speeds[0] == pytest.approx((6., 50., 432.))                  # first point: the step to the next inverted one
    assert speeds[2] is None                                            # no 2D solve at this point
    assert speeds[3] == pytest.approx((4., 4., 3600.))                  # 4 m from point 2 in 4 s: 1 m/s = 3600 m/h
    assert gif.speed_text(speeds[3]) == '2D inversion speed: 3600.0 m/h (4.00 m in 4 s)'
    assert gif.speed_text((3.048, 159., 69.01)) == '2D inversion speed:  69.0 m/h (3.05 m in 159 s)'   # fixed width
    assert gif.speed_text(None) == '2D inversion speed: n/a (no 2D solve)'
    assert gif.inversion_speeds(meta[:1], measured_depth) == {0: None}  # a single point has no step


def test_linear_color_scale_between_fixed_limits_and_the_speed_readout(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, to_rgba

    write_chain(tmp_path, map_seconds=[100., None, 400.])
    trajectory = tmp_path / 'trajectory.DAT'
    write_trajectory(trajectory)                                        # MD 10000, 10100, 10200 ft
    lwd = tmp_path / 'LWD_resistivity.txt'
    lwd.write_text('% logpt Res1 Res2\n1 2.0 4.0\n2 20.0 40.0\n3 200.0 400.0\n')
    args = gif.parse_args(['--run-dir', str(tmp_path), '--no-benchmark', '--x-axis', 'md', '--trajectory', str(trajectory),
                           '--lwd', str(lwd), '--plain-panels', '--color-scale', 'linear', '--color-limits', '0.8', '1.1',
                           '--inversion-speed', '--dpi', '25', '--output', str(tmp_path / 'out.gif')])
    frames = []
    savefig = plt.Figure.savefig
    def inspect(fig, *a, **k):
        maps, bars, profiles = fig.axes[:4], fig.axes[4:6], fig.axes[-2:]
        frames.append({'titles': [axis.get_title() for axis in maps],
                       'bars': [(bar.get_ylabel(), bar.get_ylim(), [float(tick) for tick in bar.get_yticks()]) for bar in bars],
                       'plain': [np.asarray(axis.images[0].get_array()) for axis in maps[2:]],
                       'profiles': [(axis.get_xscale(), axis.get_xlabel(), axis.get_title()) for axis in profiles],
                       'lines': [[line.get_label() for line in axis.get_lines()] for axis in profiles],
                       'dots': [[tuple(map(float, c.get_offsets()[0])) for c in axis.collections
                                 if len(c.get_offsets()) == 1 and tuple(c.get_facecolor()[0]) == to_rgba('red')]
                                for axis in profiles],
                       'speed': [text.get_text() for text in fig.texts if 'inversion speed' in text.get_text()]})
        return savefig(fig, *a, **k)
    monkeypatch.setattr(plt.Figure, 'savefig', inspect)
    output, count = gif.render(args)
    assert output.is_file() and count == 3
    last = frames[-1]
    assert last['titles'] == ['Estimated Rh (at bit), variance wash', 'Estimated Rv (at bit), variance wash',
                              'Estimated Rh (at bit), no variance wash', 'Estimated Rv (at bit), no variance wash']
    for (label, limits, ticks), key in zip(last['bars'], gif.KEYS):
        assert label == f'{key} [ohm m]' and limits == pytest.approx((.8, 1.1))    # ohm m, not log10, the given limits
        assert ticks[0] == .8 and ticks[-1] == 1.1 and ticks == sorted(ticks)      # both ends are ticked
    # The plain maps are the posterior-mean resistivity [ohm m] on the linear colour scale, clipped at its ends.
    monkeypatch.undo()
    _, collected, _, _, _ = gif.collect(args)
    colour, scale = gif.hue_colormap(), Normalize(.8, 1.1, clip=True)
    for image, key in zip(last['plain'], gif.KEYS):
        resistivity = 10. ** collected[-1]['log10'][key].T
        painted = np.isfinite(resistivity)
        beyond = (resistivity[painted] < .8) | (resistivity[painted] > 1.1)
        assert beyond.any() and not beyond.all()                                  # some cells take the end colours
        assert np.allclose(image[painted], colour(scale(resistivity[painted])))
        assert not image[~painted].any()
        assert not np.allclose(image[painted], colour(Normalize(np.log10(.8), np.log10(1.1), clip=True)(
            np.log10(resistivity[painted]))))                                      # and not the log10 colours
    # The bit-column panels are in ohm m too (logarithmic axis): the LWD dots of point 1 are 20 and 40 ohm m.
    assert last['profiles'] == [('log', f'{key} [ohm m]', f'{key}, bit column at MD = 10200 ft') for key in gif.KEYS]
    assert all('ensemble mean (geometric)' in labels for labels in last['lines'])
    assert frames[1]['dots'] == [[(pytest.approx(20.), 8001.)], [(pytest.approx(40.), 8001.)]]
    # Speed: 100 ft = 30.48 m of MD per point over the 2D solve time; point 1 had no 2D solve.
    assert [frame['speed'] for frame in frames] == [['2D inversion speed: 1097.3 m/h (30.48 m in 100 s)'],
                                                    ['2D inversion speed: n/a (no 2D solve)'],
                                                    ['2D inversion speed: 274.3 m/h (30.48 m in 400 s)']]


def test_fixed_color_limits_on_the_log10_scale(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    write_chain(tmp_path)
    args = gif.parse_args(['--run-dir', str(tmp_path), '--no-benchmark', '--color-limits', '1', '100', '--dpi', '25',
                           '--output', str(tmp_path / 'out.gif')])
    bars = []
    savefig = plt.Figure.savefig
    def inspect(fig, *a, **k):
        bars.append([(bar.get_ylabel(), bar.get_ylim(), [float(tick) for tick in bar.get_yticks()])
                     for bar in fig.axes[2:4]])                         # stacked layout: two maps, their bars, two columns
        return savefig(fig, *a, **k)
    monkeypatch.setattr(plt.Figure, 'savefig', inspect)
    gif.render(args)
    for (label, limits, ticks), key in zip(bars[-1], gif.KEYS):
        assert label == f'log10 {key} [ohm m]' and limits == pytest.approx((0., 2.))
        assert ticks[0] == 0. and ticks[-1] == 2.


def test_gif_skips_a_newest_checkpoint_that_is_still_being_written(tmp_path, capsys):
    write_chain(tmp_path, points=3)
    complete = (tmp_path / 'inversion_results_assim_2.pkl').read_bytes()
    (tmp_path / 'inversion_results_assim_3.pkl').write_bytes(complete[:len(complete) // 3])
    args = gif.parse_args(['--run-dir', str(tmp_path), '--no-benchmark', '--dpi', '25',
                           '--output', str(tmp_path / 'out.gif')])
    output, count = gif.render(args)
    assert count == 3 and 'still being written' in capsys.readouterr().out
