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
import numpy as np
from scipy.stats import norm

from plot_curtain import (AXIS_LABELS, CHECKPOINT_RE, fidelity_of, horizontal_length_map, hue_colormap,
                          label_horizontal_length, member_fields, paint_point, uncertainty_rgba)
from plot_curtain_section import FIDELITY_COLORS, draw_path, load_benchmark

KEYS = ('Rh', 'Rv')


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
                        help='LWD_resistivity.txt (log point, Res1, Res2): drawn in red in the bit-column panels, '
                             'the log so far against the trajectory TVD and the current point at the bit '
                             '(needs --trajectory)')
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
    return args


def lwd_log(lwd, trajectory):
    """log10 LWD resistivities (points x 2) and the TVD of each logging point."""
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
    if not paths:
        raise RuntimeError(f'No checkpoints in {args.run_dir}')
    meta = []
    for path in paths:
        result = load(path)
        meta.append({'index': int(CHECKPOINT_RE.search(path.name).group(1)), 'dims': result['dims'],
                     'spacing': (result['spacing_ft'][0], result['spacing_ft'][2]), 'shift': result['shift_ft'],
                     'fidelity': fidelity_of(result), 'bit': bit_position(result)})
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
    norms = {key: Normalize(*np.nanpercentile(logs[key], [2, 98]), clip=True) for key in KEYS}
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
    length_maps = horizontal_length_map(args.trajectory, x_axis) if x_axis != 'plane' else None
    position_label = AXIS_LABELS.get(x_axis, 'X [ft]')
    position_of = (lambda x: float(length_maps[0](x))) if length_maps else (lambda x: float(x))
    cmap = hue_colormap()
    white_std = np.sqrt(args.uncertainty_white_threshold * scale)
    path_x = np.array([m['bit'][0] for m in meta])
    path_z = np.array([m['bit'][1] for m in meta])
    fidelities = [m['fidelity'] for m in meta]
    counts_total = {fidelity: fidelities.count(fidelity) for fidelity in FIDELITY_COLORS}

    workdir = Path(tempfile.mkdtemp(prefix='inversion-gif-'))
    try:
        for frame_number, frame in enumerate(frames):
            number, current = frame['number'], frame['meta']
            fig = plt.figure(figsize=(15, 15))
            outer = fig.add_gridspec(2, 1, height_ratios=[2.7, 1.5], hspace=.16)
            maps_grid = outer[0].subgridspec(2, 1, hspace=.22)
            maps = [fig.add_subplot(maps_grid[row]) for row in range(2)]
            for axis, key in zip(maps, KEYS):
                rgba = uncertainty_rgba(frame['log10'][key].T, frame['variance'][key].T, cmap, norms[key], scale,
                                        args.uncertainty_strength, args.uncertainty_white_threshold)
                axis.imshow(rgba, extent=(x0, x1, z_edges[0], z_edges[-1]), origin='lower',
                            interpolation='nearest', aspect='auto')
                draw_path(axis, path_x[:number + 1], path_z[:number + 1], fidelities[:number + 1], 1.6, 4)
                axis.scatter([current['bit'][0]], [current['bit'][1]], marker='x', color='k', s=60, zorder=6)
                axis.set_xlim(x0, x1)
                axis.set_ylim(z_bottom, z_top)
                axis.set_aspect('equal', adjustable='box')
                axis.set_ylabel('TVD [ft]')
                axis.set_title(f'Estimated log10 {key} (at bit)')
                bar = fig.colorbar(ScalarMappable(norm=norms[key], cmap=cmap), ax=axis, fraction=.02, pad=.01)
                bar.set_label(f'log10 {key} [ohm m]')
                if length_maps is not None:
                    label_horizontal_length(axis, *length_maps, label=position_label)
                    axis.set_xlabel('')
            maps[1].set_xlabel(position_label)

            profile_grid = outer[1].subgridspec(1, 2, wspace=.08)
            profiles = [fig.add_subplot(profile_grid[0])]
            profiles.append(fig.add_subplot(profile_grid[1], sharey=profiles[0]))
            depth, samples = frame['depth'], frame['samples']
            bit_x, bit_z = current['bit']
            ref_column = int(np.argmin(np.abs(ref_x - bit_x))) if reference is not None else None
            position_text = f'{position_label.split(" [")[0]} = {position_of(bit_x):.0f} ft'
            for axis, key in zip(profiles, KEYS):
                low, high = np.percentile(samples[key], [1, 99], axis=0)
                mean = samples[key].mean(axis=0)
                axis.fill_betweenx(depth, low, high, step='mid', color='0.6', alpha=.5, lw=0, label='ensemble p1-p99')
                for sign in (-1, 1):
                    axis.step(mean + sign * norm.ppf(.99) * white_std, depth, where='mid', color='tab:purple',
                              ls='--', lw=1., label='white threshold (p1-p99 equiv.)' if sign > 0 else None)
                axis.step(mean, depth, where='mid', color='k', lw=1.4, label='ensemble mean')
                if reference is not None:
                    axis.plot(np.log10(reference[key][:, ref_column]), ref_z, color='red', lw=1.4, label='benchmark')
                if lwd is not None:
                    # The LWD log along the well so far (logging points 0..current) against the trajectory TVD,
                    # and the current point's two readings at the bit.
                    points = [m['index'] for m in meta[:number + 1]]
                    for column, style, label in ((0, '-', 'LWD Res1 (log so far)'), (1, '--', 'LWD Res2')):
                        axis.plot(lwd_values[points, column], lwd_tvd[points], color='red', ls=style, lw=1.,
                                  alpha=.8, label=label)
                    axis.scatter(lwd_values[current['index']], [lwd_tvd[current['index']]] * 2, color='red',
                                 marker='o', s=28, zorder=6, label='LWD at the bit')
                axis.axhline(bit_z, color=FIDELITY_COLORS[current['fidelity']], ls=':', lw=1.4, label='bit TVD')
                axis.set_xlim(*profile_limits[key])
                axis.set_ylim(z_bottom, z_top)
                axis.set_xlabel(f'log10 {key} [ohm m]')
                axis.set_title(f'log10 {key}, bit column at {position_text}', fontsize=10)
                axis.grid(alpha=.3)
            profiles[0].set_ylabel('TVD [ft]')
            profiles[1].tick_params(labelleft=False)
            profiles[0].legend(fontsize=8, loc='lower right', framealpha=.8)

            counts = {fidelity: fidelities[:number + 1].count(fidelity) for fidelity in FIDELITY_COLORS}
            handles = [Line2D([], [], color=color, lw=2.2, marker='o', ms=4, mec='k', mew=.3,
                              label=f'{fidelity} points ({counts[fidelity]} of {counts_total[fidelity]})')
                       for fidelity, color in FIDELITY_COLORS.items()]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .935), ncol=2, fontsize=10,
                       frameon=False)
            fig.suptitle(f'{args.run_dir.name}: logging point {current["index"]} ({current["fidelity"]}), '
                         f'{position_text}\nvariance wash p{args.uncertainty_percentile:g} of the final curtain, '
                         f'white above {args.uncertainty_white_threshold:g}', fontsize=12, y=.975)
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
