"""Animated evolution of a sequential inversion: one frame per logging point.

Top: the estimated log10 Rh and log10 Rv curtains as they are built point by
point, painted at the bit (nearest-bit) on the constant-lightness hue scale with
the posterior variance washed into lightness (white above the threshold), and
the well path so far coloured by the fidelity that assimilated each point (1D
blue, 2D orange). Bottom: the current point's bit column, Rh and Rv side by side:
posterior p1-p99 band, mean, benchmark and the white threshold as a p1-p99
envelope (as in ``plot_curtain_overview.py``). Window, colour bars, profile axes
and the variance scale (percentile of the final curtain) are fixed over all
frames. Checkpoints are read one at a time; the curtain is painted
incrementally exactly as ``plot_curtain.build_curtain`` does.

``--view-points N`` replaces the fixed full-curtain window by a view that moves
with the bit: N logging points wide, the bit ``--view-bit-position`` points from
its left edge, so each frame shows the estimate around the bit as it stood when
that point was inverted. ``--plain-panels`` adds the same two maps without the
variance wash under the washed ones (2 x 2: Rh column, Rv column).
``--length-unit m`` labels the horizontal and TVD axes and the positions in the
titles in metres; the geometry itself stays in feet.

With ``--trajectory`` the well path, the bit marker, the bit TVD line and the
position in the titles are the trajectory's own logging points; without it they
are the window centres, i.e. those positions snapped to the cell grid.

``--color-scale linear`` colours the maps by resistivity in ohm m instead of its
log10, and ``--color-limits MIN MAX`` fixes the two ends of the colour scale for
Rh and Rv alike (values outside take the end colours). With the linear scale the
bit-column panels are in ohm m too, on a logarithmic axis. ``--inversion-speed``
prints, for every logging point, how fast its 2D inversion advanced along the
well: the MD step from the previous logging point over the time its 2D MAP took
to solve, in m/h.
"""

import argparse
import pickle
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoLocator, FuncFormatter, Locator
import numpy as np
from scipy.stats import norm

from field_trajectory import FEET_PER_METER
from plot_curtain import (AXIS_LABELS, CHECKPOINT_RE, fidelity_of, horizontal_length_map, hue_colormap,
                          label_horizontal_length, member_fields, paint_point, uncertainty_rgba)
from plot_curtain_section import FIDELITY_COLORS, draw_path, load_benchmark

KEYS = ('Rh', 'Rv')
UNPAINTED = '0.86'      # moving view: background of cells that hold no model yet
FEET_PER_UNIT = {'ft': 1., 'm': FEET_PER_METER}     # --length-unit: the geometry is in feet, only the labels change
POSITION_DECIMALS = {'ft': 0, 'm': 1}               # decimals of the bit position in the titles


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--benchmark', type=Path, default=here.parent / 'Dashboard/benchmarks/Benchmark-3')
    reference.add_argument('--no-benchmark', action='store_const', const=None, dest='benchmark',
                           help='Field data: no reference model; colour scales come from the final curtain')
    parser.add_argument('--x-axis', choices=['plane', 'horizontal-length', 'thl', 'md'], default='plane',
                        help="Horizontal axis labels (see plot_curtain.py); the geometry stays in the curtain plane")
    parser.add_argument('--trajectory', type=Path, default=None,
                        help='Field trajectory file for --x-axis horizontal-length / thl / md and --lwd')
    parser.add_argument('--lwd', type=Path, default=None,
                        help='LWD_resistivity.txt (log point, Res1 = LWD Rh, Res2 = LWD Rv): the current point\'s '
                             'reading is drawn as a red dot at the bit TVD in the Rh and Rv bit-column panels '
                             '(needs --trajectory)')
    parser.add_argument('--view-points', type=int, default=None, metavar='N',
                        help='Moving view: show N logging points around the bit instead of the whole curtain '
                             '(the positions come from --trajectory when given, else from the checkpoints)')
    parser.add_argument('--view-bit-position', type=int, default=None, metavar='K',
                        help='Logging points between the left edge of the moving view and the bit (K behind, '
                             'N - K ahead; default three quarters of N)')
    parser.add_argument('--plain-panels', action='store_true',
                        help='Add the Rh and Rv maps without the variance wash under the washed ones '
                             '(2 x 2: Rh column, Rv column)')
    parser.add_argument('--length-unit', choices=list(FEET_PER_UNIT), default='ft',
                        help='Unit of the horizontal (X / MD / THL) and TVD axes and of the bit position in the '
                             'titles; the inputs and the geometry are in feet either way')
    parser.add_argument('--color-scale', choices=['log10', 'linear'], default='log10',
                        help='Colour the maps by log10 resistivity or linearly by resistivity [ohm m]; with linear '
                             'the bit-column panels are in ohm m as well, on a logarithmic axis')
    parser.add_argument('--color-limits', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help='Resistivity [ohm m] at the two ends of the colour scale, for Rh and Rv and all frames '
                             '(default: 2nd and 98th percentile of the benchmark, or of the final curtain)')
    parser.add_argument('--inversion-speed', action='store_true',
                        help='Print the 2D inversion speed of every logging point [m/h]: its MD step from the '
                             'previous logging point over the time its 2D MAP took to solve (needs --trajectory)')
    parser.add_argument('--uncertainty-percentile', type=float, default=95.)
    parser.add_argument('--uncertainty-strength', type=float, default=.75)
    parser.add_argument('--uncertainty-white-threshold', type=float, default=.35)
    parser.add_argument('--every', type=int, default=1, help='Use every n-th logging point as a frame')
    parser.add_argument('--frame-ms', type=int, default=120, help='GIF frame duration')
    parser.add_argument('--dpi', type=int, default=70)
    parser.add_argument('--output', type=Path, default=None, help='Default <run-dir>/inversion-evolution.gif')
    parser.add_argument('--mp4', action='store_true', help='Also write an MP4 next to the GIF (ffmpeg)')
    args = parser.parse_args(argv)
    if args.x_axis != 'plane' and args.trajectory is None:
        parser.error(f'--x-axis {args.x_axis} needs --trajectory')
    if args.lwd is not None and args.trajectory is None:
        parser.error('--lwd needs --trajectory (TVD of the logging points)')
    if args.inversion_speed and args.trajectory is None:
        parser.error('--inversion-speed needs --trajectory (MD of the logging points)')
    if args.color_limits is not None:
        low, high = args.color_limits
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            parser.error('--color-limits must be two increasing resistivities')
        if args.color_scale == 'log10' and low <= 0:
            parser.error('--color-limits must be positive on the log10 colour scale')
    if args.view_points is None:
        if args.view_bit_position is not None:
            parser.error('--view-bit-position needs --view-points')
    else:
        if args.view_points < 2:
            parser.error('--view-points must be at least 2')
        if args.view_bit_position is None:
            args.view_bit_position = int(round(.75 * args.view_points))
        if not 0 <= args.view_bit_position <= args.view_points:
            parser.error('--view-bit-position must lie between 0 and --view-points')
    return args


def trajectory_positions(trajectory):
    """Along-plane x and TVD [ft] of every logging point of a field trajectory."""
    from field_trajectory import along_well_trajectory, read_trajectory
    values = read_trajectory(trajectory)
    well, _ = along_well_trajectory(values)
    return well['X'][:, 0] * FEET_PER_METER, values[:, 1].copy()


def logging_positions(trajectory, meta):
    """Along-plane x [ft] of the logging points by index.

    The trajectory gives every logging point, also those not yet inverted; without one the
    checkpoints' window centres are used (interpolated over missing indices).
    """
    if trajectory is not None:
        return trajectory_positions(trajectory)[0]
    indices = np.array([m['index'] for m in meta])
    if len(indices) < 2:
        raise ValueError('A moving view needs at least two logging points')
    return np.interp(np.arange(indices.max() + 1), indices, [m['bit'][0] for m in meta])


def view_limits(positions, index, points, bit_position):
    """Plane-x limits of a view ``points`` logging points wide with the bit ``bit_position`` points from the left.

    Logging points before the first or after the last one are extrapolated with the end spacing,
    so the bit keeps its place in the view at both ends of the well.
    """
    positions = np.asarray(positions, dtype=float)
    last = len(positions) - 1

    def at(point):
        if point < 0:
            return positions[0] + point * (positions[1] - positions[0])
        if point > last:
            return positions[last] + (point - last) * (positions[last] - positions[last - 1])
        return positions[point]

    return float(at(index - bit_position)), float(at(index + points - bit_position))


def view_columns(x_edges, limits):
    """Canvas column range [first, last) that intersects the view; empty when it lies outside the canvas."""
    dx = x_edges[1] - x_edges[0]
    columns = len(x_edges) - 1
    first = int(np.clip(np.floor((limits[0] - x_edges[0]) / dx), 0, columns))
    last = int(np.clip(np.ceil((limits[1] - x_edges[0]) / dx), 0, columns))
    return first, last


class UnitLocator(Locator):
    """Round tick values of a display unit on an axis whose coordinates are in feet."""

    def __init__(self, feet_per_unit):
        self.feet_per_unit = feet_per_unit
        self.rounded = AutoLocator()

    def set_axis(self, axis):
        super().set_axis(axis)
        self.rounded.set_axis(axis)         # as many ticks as fit the axis, like the default locator

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        low, high = sorted((vmin, vmax))
        return self.rounded.tick_values(low / self.feet_per_unit, high / self.feet_per_unit) * self.feet_per_unit


def tick_in_unit(axis, feet_per_unit):
    """Tick a feet axis (``axes.xaxis`` or ``axes.yaxis``) at round values of the display unit."""
    if feet_per_unit != 1.:
        axis.set_major_locator(UnitLocator(feet_per_unit))
        axis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / feet_per_unit:g}'))


def maps_in_unit(length_maps, feet_per_unit):
    """``horizontal_length_map``'s pair with the length in the display unit (plane x stays in feet)."""
    to_length, to_plane = length_maps
    if feet_per_unit == 1.:
        return to_length, to_plane

    def unit_length(plane_x):
        return to_length(plane_x) / feet_per_unit

    def unit_plane(length):
        return to_plane(np.asarray(length, dtype=float) * feet_per_unit)

    unit_length.monotone = getattr(to_length, 'monotone', True)
    return unit_length, unit_plane


def limit_ticks(bar, norm_):
    """Tick both ends of a colour bar with fixed limits; its own ticks too close to an end make way."""
    low, high = norm_.vmin, norm_.vmax
    margin = .05 * (high - low)
    inner = [tick for tick in bar.get_ticks() if low + margin < tick < high - margin]
    bar.set_ticks([low, *inner, high])


def map_colorbar(fig, axes, norm_, cmap, label, fixed=False):
    """One colour bar to the right of a column of equal-aspect maps, aligned with their drawn boxes.

    ``fixed``: the limits were chosen by the user, so both ends are ticked and drawn as arrows
    (values beyond them take the end colours).
    """
    for axis in axes:
        axis.apply_aspect()
    boxes = [axis.get_position() for axis in axes]
    right = max(box.x1 for box in boxes)
    bottom, top = min(box.y0 for box in boxes), max(box.y1 for box in boxes)
    bar = fig.colorbar(ScalarMappable(norm=norm_, cmap=cmap), cax=fig.add_axes([right + .008, bottom, .011, top - bottom]),
                       extend='both' if fixed else 'neither')
    if fixed:
        limit_ticks(bar, norm_)
    bar.set_label(label)
    return bar


def inversion_speeds(meta, measured_depth_m):
    """2D inversion speed of every inverted logging point: {index: (MD step [m], solve time [s], speed [m/h])}.

    The step is the MD from the previously inverted logging point (for the first point: to the next
    one), the time is the 2D MAP solve time recorded in the checkpoint. Points without a 2D solve
    (1D only) or without a neighbour map to None.
    """
    speeds = {}
    for place, point in enumerate(meta):
        neighbour = meta[place - 1] if place else (meta[1] if len(meta) > 1 else None)
        seconds = point.get('map_seconds')
        if neighbour is None or seconds is None or not seconds > 0:
            speeds[point['index']] = None
            continue
        step = abs(float(measured_depth_m[point['index']] - measured_depth_m[neighbour['index']]))
        speeds[point['index']] = (step, float(seconds), 3600. * step / float(seconds))
    return speeds


def speed_text(speed):
    if speed is None:
        return '2D inversion speed: n/a (no 2D solve)'
    step, seconds, value = speed
    return f'2D inversion speed: {value:5.1f} m/h ({step:.2f} m in {seconds:.0f} s)'


def lwd_log(lwd, trajectory):
    """log10 LWD resistivities (points x 2: Rh, Rv) and the TVD of each logging point."""
    from field_trajectory import read_trajectory
    values = np.loadtxt(lwd, comments='%', ndmin=2)
    tvd = read_trajectory(trajectory)[:, 1]
    if values.shape != (len(tvd), 3) or np.any(values[:, 1:] <= 0):
        raise ValueError('LWD rows must match the trajectory logging points with positive resistivities')
    return np.log10(values[:, 1:]), tvd


def checkpoint_paths(run_dir):
    paths = {}
    for path in run_dir.glob('inversion_results_assim_*.pkl'):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            paths[int(match.group(1))] = path
    return [paths[index] for index in sorted(paths)]


def load(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def readable_paths(paths):
    """Drop the newest checkpoint if it cannot be read yet (a running chain may still be writing it)."""
    try:
        load(paths[-1])
    except (EOFError, pickle.UnpicklingError):
        print(f'skipping {paths[-1].name}: still being written')
        return paths[:-1]
    return paths


def bit_position(result):
    dx, dz = result['spacing_ft'][0], result['spacing_ft'][2]
    return (result['shift_ft']['x'] + result['dims'][0] / 2 * dx, result['shift_ft']['z'] + result['dims'][2] / 2 * dz)


def column_samples(result):
    """log10 Rh / Rv members of the point's bit column and their cell-centre TVD."""
    nx, _, nz = result['dims']
    members = [member_fields(params, result['dims'], result.get('rv_ratio_bounds', (1., 5.)),
                             result.get('rv_hard_cap_ohmm'), result.get('anisotropy', 'probit'))
               for params in result.get('posterior_ensemble_params', result['posterior_params'])]
    column = (lambda field: field) if not members[0][2] else (lambda field: field[nx // 2])
    samples = {key: np.array([column(member[slot]) for member in members]) / np.log(10.)
               for slot, key in ((0, 'Rh'), (1, 'Rv'))}
    depth = result['shift_ft']['z'] + (np.arange(nz) + .5) * result['spacing_ft'][2]
    return samples, depth


def collect(args):
    """Pass 1: window extent and path; pass 2: incremental at-bit painting, one snapshot per frame."""
    paths = checkpoint_paths(args.run_dir)
    if paths:
        paths = readable_paths(paths)
    if not paths:
        raise RuntimeError(f'No checkpoints in {args.run_dir}')
    meta = []
    for path in paths:
        result = load(path)
        meta.append({'index': int(CHECKPOINT_RE.search(path.name).group(1)), 'dims': result['dims'],
                     'spacing': (result['spacing_ft'][0], result['spacing_ft'][2]), 'shift': result['shift_ft'],
                     'fidelity': fidelity_of(result), 'bit': bit_position(result),
                     'map_seconds': (result.get('laplace') or {}).get('map_seconds')})
    spacing = meta[0]['spacing']
    origin = (min(m['shift']['x'] for m in meta), min(m['shift']['z'] for m in meta))
    n_x = int(round((max(m['shift']['x'] + m['dims'][0] * spacing[0] for m in meta) - origin[0]) / spacing[0]))
    n_z = int(round((max(m['shift']['z'] + m['dims'][2] * spacing[1] for m in meta) - origin[1]) / spacing[1]))
    canvas = {'Rh': np.full((n_x, n_z), np.nan), 'Rv': np.full((n_x, n_z), np.nan)}
    variance = {'Rh': np.full((n_x, n_z), np.nan), 'Rv': np.full((n_x, n_z), np.nan)}
    coverage, distance = np.zeros((n_x, n_z)), np.full((n_x, n_z), np.inf)
    frames = []
    for number, path in enumerate(paths):
        result = load(path)
        paint_point(canvas['Rh'], canvas['Rv'], coverage, result, origin, spacing,
                    variance['Rh'], variance['Rv'], distance)
        if number % args.every and number != len(paths) - 1:
            continue
        samples, depth = column_samples(result)
        frames.append({'number': number, 'meta': meta[number],
                       'log10': {key: np.log10(np.exp(canvas[key])).copy() for key in KEYS},
                       'variance': {key: variance[key].copy() for key in KEYS},
                       'samples': samples, 'depth': depth})
        print(f'painted {meta[number]["index"]}', end='\r', flush=True)
    print()
    finite = np.concatenate([variance[key][np.isfinite(variance[key])] for key in KEYS])
    scale = float(np.percentile(finite, args.uncertainty_percentile))
    x_edges = origin[0] + np.arange(n_x + 1) * spacing[0]
    z_edges = origin[1] + np.arange(n_z + 1) * spacing[1]
    return meta, frames, scale, x_edges, z_edges


def render(args):
    meta, frames, scale, x_edges, z_edges = collect(args)
    if getattr(args, 'trajectory', None) is not None:
        # Field data: draw the bit where the trajectory puts it. The window centre the checkpoints give
        # is that position snapped to the cell grid (half a cell along the well, up to a cell in TVD).
        well_x, well_z = trajectory_positions(args.trajectory)
        if max(point['index'] for point in meta) >= len(well_x):
            raise ValueError('The trajectory has fewer logging points than the chain')
        for point in meta:
            point['bit'] = (float(well_x[point['index']]), float(well_z[point['index']]))
    x0, x1, z_top, z_bottom = x_edges[0], x_edges[-1], z_edges[0], z_edges[-1]
    if args.benchmark:
        reference, ref_x_edges, ref_z_edges = load_benchmark(args.benchmark)
        ref_x = .5 * (ref_x_edges[1:] + ref_x_edges[:-1])
        ref_z = .5 * (ref_z_edges[1:] + ref_z_edges[:-1])
        inside = np.ix_((ref_z >= z_top) & (ref_z <= z_bottom), (ref_x >= x0) & (ref_x <= x1))
        logs = {key: np.log10(reference[key][inside]) for key in KEYS}
    else:
        # Field data: the final curtain sets the colour scale, the bit columns set the profile axes.
        reference = None
        logs = {key: frames[-1]['log10'][key][np.isfinite(frames[-1]['log10'][key])] for key in KEYS}
    # Everything is held in log10 ohm m; a linear colour scale shows it in ohm m.
    linear = getattr(args, 'color_scale', 'log10') == 'linear'
    color_limits = getattr(args, 'color_limits', None)
    in_scale = ((lambda log10_values: 10. ** np.asarray(log10_values, dtype=float)) if linear
                else (lambda log10_values: log10_values))
    quantity = {key: key if linear else f'log10 {key}' for key in KEYS}
    if color_limits is not None:
        ends = tuple(color_limits) if linear else tuple(np.log10(color_limits))
        norms = {key: Normalize(*ends, clip=True) for key in KEYS}
    else:
        norms = {key: Normalize(*np.nanpercentile(in_scale(logs[key]), [2, 98]), clip=True) for key in KEYS}
    mean_label = 'ensemble mean (geometric)' if linear else 'ensemble mean'
    if reference is not None:
        profile_limits = {key: (float(np.nanmin(logs[key])) - .8, float(np.nanmax(logs[key])) + .6) for key in KEYS}
    else:
        profile_limits = {key: (float(min(np.percentile(f['samples'][key], 1) for f in frames)) - .3,
                                float(max(np.percentile(f['samples'][key], 99) for f in frames)) + .3) for key in KEYS}
    lwd = lwd_log(args.lwd, args.trajectory) if getattr(args, 'lwd', None) else None
    if lwd is not None:
        lwd_values, lwd_tvd = lwd
        profile_limits = {key: (min(limits[0], float(lwd_values.min()) - .1), max(limits[1], float(lwd_values.max()) + .1))
                          for key, limits in profile_limits.items()}
    x_axis = getattr(args, 'x_axis', 'plane')
    unit = getattr(args, 'length_unit', 'ft')
    feet = FEET_PER_UNIT[unit]
    length_maps = (maps_in_unit(horizontal_length_map(args.trajectory, x_axis), feet)
                   if x_axis != 'plane' else None)
    position_label = AXIS_LABELS.get(x_axis, 'X [ft]').replace('[ft]', f'[{unit}]')
    depth_label = f'TVD [{unit}]'
    position_of = (lambda x: float(length_maps[0](x))) if length_maps else (lambda x: float(x) / feet)
    cmap = hue_colormap()
    white_std = np.sqrt(args.uncertainty_white_threshold * scale)
    path_x = np.array([m['bit'][0] for m in meta])
    path_z = np.array([m['bit'][1] for m in meta])
    fidelities = [m['fidelity'] for m in meta]
    counts_total = {fidelity: fidelities.count(fidelity) for fidelity in FIDELITY_COLORS}
    view = getattr(args, 'view_points', None) is not None
    plain = bool(getattr(args, 'plain_panels', False))
    positions = logging_positions(getattr(args, 'trajectory', None), meta) if view else None
    view_text = (f'view of {args.view_points} logging points: {args.view_bit_position} behind the bit, '
                 f'{args.view_points - args.view_bit_position} ahead; ' if view else '')
    speeds = None
    if getattr(args, 'inversion_speed', False):
        from field_trajectory import read_trajectory
        speeds = inversion_speeds(meta, read_trajectory(args.trajectory)[:, 2] / FEET_PER_METER)

    workdir = Path(tempfile.mkdtemp(prefix='inversion-gif-'))
    try:
        for frame_number, frame in enumerate(frames):
            number, current = frame['number'], frame['meta']
            limits = (view_limits(positions, current['index'], args.view_points, args.view_bit_position)
                      if view else (x0, x1))
            first, last = view_columns(x_edges, limits)

            def draw_map(axis, key, washed):
                """One estimate map up to this frame, with or without the variance wash."""
                if last > first:
                    rgba = uncertainty_rgba(in_scale(frame['log10'][key][first:last]).T,
                                            frame['variance'][key][first:last].T,
                                            cmap, norms[key], scale if washed else 0.,
                                            args.uncertainty_strength, args.uncertainty_white_threshold)
                    axis.imshow(rgba, extent=(x_edges[first], x_edges[last], z_edges[0], z_edges[-1]),
                                origin='lower', interpolation='nearest', aspect='auto')
                draw_path(axis, path_x[:number + 1], path_z[:number + 1], fidelities[:number + 1], 1.6, 4)
                axis.scatter([current['bit'][0]], [current['bit'][1]], marker='x', color='k', s=60, zorder=6)
                axis.set_xlim(*limits)
                axis.set_ylim(z_bottom, z_top)
                axis.set_aspect('equal', adjustable='box')
                if view:
                    axis.set_facecolor(UNPAINTED)       # no model there yet; white is the variance threshold
                if length_maps is not None:
                    label_horizontal_length(axis, *length_maps, label=position_label)
                    axis.set_xlabel('')
                else:
                    tick_in_unit(axis.xaxis, feet)
                tick_in_unit(axis.yaxis, feet)

            if plain:
                # 2 x 2: Rh column and Rv column, the washed maps over the plain ones.
                fig = plt.figure(figsize=(15, 13))
                outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.], hspace=.2, left=.065, right=.945,
                                         top=.875, bottom=.055)
                maps_grid = outer[0].subgridspec(2, 2, hspace=.12, wspace=.2)
                maps = [fig.add_subplot(maps_grid[row, column]) for row in range(2) for column in range(2)]
                for place, axis in enumerate(maps):
                    row, column = divmod(place, 2)
                    draw_map(axis, KEYS[column], washed=row == 0)
                    axis.set_title(f'Estimated {quantity[KEYS[column]]} (at bit), '
                                   + ('variance wash' if row == 0 else 'no variance wash'), fontsize=11)
                    if column == 0:
                        axis.set_ylabel(depth_label)
                    else:
                        axis.tick_params(labelleft=False)
                    if row == 0:
                        axis.tick_params(labelbottom=False)
                    else:
                        axis.set_xlabel(position_label)
                for column, key in enumerate(KEYS):
                    map_colorbar(fig, maps[column::2], norms[key], cmap, f'{quantity[key]} [ohm m]',
                                 fixed=color_limits is not None)
            else:
                fig = plt.figure(figsize=(15, 15))
                outer = fig.add_gridspec(2, 1, height_ratios=[2.7, 1.5], hspace=.16)
                maps_grid = outer[0].subgridspec(2, 1, hspace=.22)
                maps = [fig.add_subplot(maps_grid[row]) for row in range(2)]
                for axis, key in zip(maps, KEYS):
                    draw_map(axis, key, washed=True)
                    axis.set_ylabel(depth_label)
                    axis.set_title(f'Estimated {quantity[key]} (at bit)')
                    bar = fig.colorbar(ScalarMappable(norm=norms[key], cmap=cmap), ax=axis, fraction=.02, pad=.01,
                                       extend='both' if color_limits is not None else 'neither')
                    if color_limits is not None:
                        limit_ticks(bar, norms[key])
                    bar.set_label(f'{quantity[key]} [ohm m]')
                maps[1].set_xlabel(position_label)

            profile_grid = outer[1].subgridspec(1, 2, wspace=.08)
            profiles = [fig.add_subplot(profile_grid[0])]
            profiles.append(fig.add_subplot(profile_grid[1], sharey=profiles[0]))
            depth, samples = frame['depth'], frame['samples']
            bit_x, bit_z = current['bit']
            ref_column = int(np.argmin(np.abs(ref_x - bit_x))) if reference is not None else None
            position_text = (f'{position_label.split(" [")[0]} = '
                             f'{position_of(bit_x):.{POSITION_DECIMALS[unit]}f} {unit}')
            for axis, key in zip(profiles, KEYS):
                low, high = np.percentile(samples[key], [1, 99], axis=0)
                mean = samples[key].mean(axis=0)
                if linear:
                    # Resistivity in ohm m; the band and the LWD readings span decades, so the axis is logarithmic.
                    axis.set_xscale('log')
                    axis.xaxis.set_major_formatter(FuncFormatter(
                        lambda value, _: f'{value:.0f}' if value >= 1 else f'{value:g}'))
                axis.fill_betweenx(depth, in_scale(low), in_scale(high), step='mid', color='0.6', alpha=.5, lw=0,
                                   label='ensemble p1-p99')
                for sign in (-1, 1):
                    axis.step(in_scale(mean + sign * norm.ppf(.99) * white_std), depth, where='mid',
                              color='tab:purple', ls='--', lw=1.,
                              label='white threshold (p1-p99 equiv.)' if sign > 0 else None)
                axis.step(in_scale(mean), depth, where='mid', color='k', lw=1.4, label=mean_label)
                if reference is not None:
                    axis.plot(in_scale(np.log10(reference[key][:, ref_column])), ref_z, color='red', lw=1.4,
                              label='benchmark')
                if lwd is not None:
                    # The LWD reading at the bit: Res1 is the LWD Rh, Res2 the LWD Rv.
                    axis.scatter([float(in_scale(lwd_values[current['index'], KEYS.index(key)]))],
                                 [lwd_tvd[current['index']]],
                                 color='red', marker='o', s=36, zorder=6, label=f'LWD {key} at the bit')
                axis.axhline(bit_z, color=FIDELITY_COLORS[current['fidelity']], ls=':', lw=1.4, label='bit TVD')
                axis.set_xlim(*in_scale(profile_limits[key]))
                axis.set_ylim(z_bottom, z_top)
                axis.set_xlabel(f'{quantity[key]} [ohm m]')
                axis.set_title(f'{quantity[key]}, bit column at {position_text}', fontsize=10)
                axis.grid(alpha=.3)
            tick_in_unit(profiles[0].yaxis, feet)       # the two panels share the TVD axis and its ticker
            profiles[0].set_ylabel(depth_label)
            profiles[1].tick_params(labelleft=False)
            profiles[0].legend(fontsize=8, loc='lower right', framealpha=.8)

            counts = {fidelity: fidelities[:number + 1].count(fidelity) for fidelity in FIDELITY_COLORS}
            handles = [Line2D([], [], color=color, lw=2.2, marker='o', ms=4, mec='k', mew=.3,
                              label=f'{fidelity} points ({counts[fidelity]} of {counts_total[fidelity]})')
                       for fidelity, color in FIDELITY_COLORS.items()]
            if view:
                handles.append(Patch(facecolor=UNPAINTED, edgecolor='0.6', label='not yet inverted'))
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .935), ncol=len(handles),
                       fontsize=10, frameon=False)
            fig.suptitle(f'{args.run_dir.name}: logging point {current["index"]} ({current["fidelity"]}), '
                         f'{position_text}\n{view_text}variance wash p{args.uncertainty_percentile:g} of the final '
                         f'curtain, white above {args.uncertainty_white_threshold:g}', fontsize=12, y=.975)
            if speeds is not None:
                # The speed readout sits in the top right corner, above the second title line.
                fig.text(.988, .993, speed_text(speeds[current['index']]), ha='right', va='top', fontsize=11,
                         family='monospace', bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white',
                                                   'edgecolor': '0.5'})
            fig.savefig(workdir / f'frame-{frame_number:04d}.png', dpi=args.dpi)
            plt.close(fig)
            print(f'frame {frame_number + 1}/{len(frames)}', end='\r', flush=True)
        print()
        from PIL import Image
        images = [Image.open(path) for path in sorted(workdir.glob('frame-*.png'))]
        size = images[0].size
        images = [image.convert('RGB').resize(size) for image in images]
        palette = [image.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for image in images]
        output = args.output or args.run_dir / 'inversion-evolution.gif'
        durations = [args.frame_ms] * (len(palette) - 1) + [3000]      # hold the final state
        palette[0].save(output, save_all=True, append_images=palette[1:], duration=durations, loop=0,
                        optimize=False, disposal=1)
        if args.mp4 and shutil.which('ffmpeg'):
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(round(1000 / args.frame_ms)),
                            '-i', str(workdir / 'frame-%04d.png'), '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                            '-pix_fmt', 'yuv420p', str(output.with_suffix('.mp4'))], check=True)
        return output, len(frames)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None):
    output, count = render(parse_args(argv))
    print(f'{count} frames -> {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
