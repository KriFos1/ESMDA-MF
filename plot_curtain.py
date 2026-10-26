"""Gradually build the inversion curtain from sequential checkpoints.

Reads ``inversion_results_assim_<index>.pkl`` files from a run directory,
places every local-window posterior at its world position, and renders the
estimated Rh/Rv curtains next to the benchmark reference, with the well
trajectory and the per-point measurement misfit. ``--follow`` re-renders
whenever new checkpoints appear, so the curtain grows during a run.
"""

import argparse
import colorsys
import pickle
import re
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import norm

from run_DA_sequential import meter_to_feet

CHECKPOINT_RE = re.compile(r"inversion_results_assim_(\d+)\.pkl$")
HUE_RANGE = (240., 0.)          # degrees: blue (conductive) to red (resistive)


def hue_colormap(samples=256):
    """Constant-lightness hue scale: HLS lightness 0.5 and full saturation, blue to red.

    The value lives only in the hue, so ``hls_lightness_rgb`` can encode the
    posterior variance in lightness without shifting the apparent value.
    """
    hues = np.linspace(*HUE_RANGE, samples) / 360.
    return ListedColormap([colorsys.hls_to_rgb(hue, .5, 1.) for hue in hues], name='hls_hue')


def value_colormap(name):
    return hue_colormap() if name == 'hue' else plt.get_cmap(name)


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Sequential run output directory")
    parser.add_argument("--benchmark", type=Path, default=here.parent / "Dashboard/benchmarks/Benchmark-3",
                        help="Benchmark root for the reference curtain (blank string disables)")
    parser.add_argument("--output", type=Path, default=None, help="Output image (default <run-dir>/curtain.png)")
    parser.add_argument("--margin-ft", type=float, default=50., help="Curtain margin around the well")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument('--log10-limits', type=float, nargs=2, default=None)
    parser.add_argument('--shared-extents', action='store_true')
    parser.add_argument('--estimate-label', default='Estimated')
    parser.add_argument('--colormap', choices=['viridis', 'hue'], default='viridis',
                        help="Value colours of all panels: 'viridis', or 'hue' (constant HLS lightness, "
                             'blue to red), which leaves lightness free for the uncertainty wash')
    parser.add_argument('--uncertainty-lightness', action='store_true',
                        help='Map posterior field variance into estimate-color lightness')
    parser.add_argument('--uncertainty-percentile', type=float, default=95.,
                        help='Per-render percentile used to normalize ensemble variance')
    parser.add_argument('--uncertainty-strength', type=float, default=.75,
                        help='Maximum HLS lightness wash, from 0 (off) to 1 (white)')
    parser.add_argument('--uncertainty-white-threshold', type=float, default=.35,
                        help='Set cells above this normalized variance to pure white')
    parser.add_argument('--paint', choices=['last', 'nearest-bit'], default='last',
                        help='last: later windows overwrite earlier ones; nearest-bit: each cell comes from '
                             'the window in which it was laterally closest to the bit (at-drilling estimate)')
    parser.add_argument("--follow", action="store_true", help="Keep re-rendering as checkpoints appear")
    parser.add_argument("--interval", type=float, default=60., help="Follow-mode poll interval, seconds")
    args = parser.parse_args(argv)
    if (args.dpi < 20 or args.interval <= 0 or args.margin_ft < 0
            or not 0 < args.uncertainty_percentile <= 100
            or not 0 <= args.uncertainty_strength <= 1
            or not 0 <= args.uncertainty_white_threshold <= 1):
        parser.error("Require dpi >= 20, interval > 0, margin >= 0, percentile in (0, 100] "
                     "and uncertainty strength/white threshold in [0, 1]")
    return args


# Checkpoint entries the curtain uses; the other ensembles (prior, post-MDA, ...) are
# dropped at load time, so memory grows by one ensemble per point, not four.
RENDER_KEYS = ('dims', 'spacing_ft', 'shift_ft', 'posterior_ensemble_params', 'posterior_params',
               'rv_ratio_bounds', 'rv_hard_cap_ohmm', 'anisotropy', 'selected_fidelity', 'optimized_2d',
               'posterior_prediction_kind', 'two_d_mode', 'misfit', 'posterior_misfit',
               'best_member_data_misfit', 'misfit_threshold')


def trim_checkpoint(result):
    """Keep only what ``render`` reads, with a single posterior ensemble."""
    trimmed = {key: result[key] for key in RENDER_KEYS if key in result}
    if 'posterior_ensemble_params' in trimmed:
        trimmed['posterior_params'] = trimmed['posterior_ensemble_params']
    return trimmed


def load_checkpoints(run_dir, cache=None):
    """Trimmed checkpoints in index order; ``cache`` (a dict) reuses unchanged files across calls."""
    checkpoints = []
    for path in sorted(run_dir.glob("inversion_results_assim_*.pkl")):
        match = CHECKPOINT_RE.search(path.name)
        if match is None:
            continue
        stat = path.stat()
        key = (stat.st_size, stat.st_mtime_ns)
        if cache is not None and path in cache and cache[path][0] == key:
            checkpoints.append((int(match.group(1)), cache[path][1]))
            continue
        try:
            with path.open("rb") as stream:
                result = trim_checkpoint(pickle.load(stream))
        except (pickle.UnpicklingError, EOFError, AttributeError):
            continue
        ensemble_params = result.get('posterior_ensemble_params', result['posterior_params'])
        if not np.isfinite(np.asarray(ensemble_params, dtype=float)).all():
            continue
        if cache is not None:
            cache[path] = (key, result)
        checkpoints.append((int(match.group(1)), result))
    return sorted(checkpoints, key=lambda item: item[0])


def fidelity_of(result):
    fidelity = result.get("selected_fidelity")
    if fidelity is None:
        fidelity = "2D" if result.get("optimized_2d", True) else "1D"
    return fidelity


def prediction_kind(result):
    return result.get('posterior_prediction_kind',
                      'linearized' if result.get('two_d_mode') == 'single-update' else 'forward')


def member_fields(params, dims, ratio_bounds=(1., 5.), rv_cap_ohmm=None, anisotropy='probit'):
    """One member's (ln Rh, ln Rv) on the window; 1D members return columns."""
    from esmda_core import log_ratio_from_latent
    nx, _, nz = dims
    params = np.asarray(params, dtype=float)
    n = nx * nz
    if params.size == 2 * n:
        log_rh = params[:n].reshape(nx, nz)
        latent = params[n:].reshape(nx, nz)
        ratio_min, ratio_max = ratio_bounds
        log_rv = log_rh + log_ratio_from_latent(latent, ratio_min, ratio_max, anisotropy)
        if rv_cap_ohmm is not None:
            log_rv = np.minimum(log_rv, np.log(rv_cap_ohmm))
        return log_rh, log_rv, True
    log_rh = params[:nz]
    latent = params[nz:]
    ratio_min, ratio_max = ratio_bounds
    log_rv = log_rh + log_ratio_from_latent(latent, ratio_min, ratio_max, anisotropy)
    if rv_cap_ohmm is not None:
        log_rv = np.minimum(log_rv, np.log(rv_cap_ohmm))
    return log_rh, log_rv, False


def posterior_moments(params, dims, ratio_bounds=(1., 5.), rv_cap_ohmm=None, anisotropy='probit'):
    """Return posterior means and sample variances in log10 resistivity units."""
    members = [member_fields(param, dims, ratio_bounds, rv_cap_ohmm, anisotropy) for param in params]
    rh_samples = np.asarray([member[0] for member in members], dtype=float)
    rv_samples = np.asarray([member[1] for member in members], dtype=float)
    if rh_samples.shape[0] > 1:
        rh_variance = np.var(rh_samples / np.log(10.), axis=0, ddof=1)
        rv_variance = np.var(rv_samples / np.log(10.), axis=0, ddof=1)
    else:
        rh_variance = np.zeros_like(rh_samples[0])
        rv_variance = np.zeros_like(rv_samples[0])
    return (rh_samples.mean(axis=0), rv_samples.mean(axis=0),
            rh_variance, rv_variance, members[0][2])


def hls_lightness_rgb(rgb, normalized_variance, strength=.75, white_threshold=.35):
    """Preserve hue/saturation while washing uncertain colors toward white."""
    level = float(np.clip(normalized_variance, 0., 1.))
    if level > white_threshold:
        return 1., 1., 1.
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    lightness += strength * level * (1. - lightness)
    return colorsys.hls_to_rgb(hue, min(lightness, 1.), saturation)


def uncertainty_rgba(field, variance, cmap, norm, variance_scale, strength=.75,
                     white_threshold=.35):
    """Map a mean field to its value colors, then encode variance in HLS lightness."""
    field = np.asarray(field, dtype=float)
    variance = np.asarray(variance, dtype=float)
    rgba = np.zeros((*field.shape, 4), dtype=float)
    valid_field = np.isfinite(field)
    rgba[valid_field] = cmap(norm(field[valid_field]))
    valid_variance = valid_field & np.isfinite(variance) & (variance >= 0.)
    if variance_scale > 0 and strength > 0:
        normalized = np.clip(variance[valid_variance] / variance_scale, 0., 1.)
        flat_rgba = rgba.reshape(-1, 4)
        for index, level in zip(np.flatnonzero(valid_variance), normalized):
            flat_rgba[index, :3] = hls_lightness_rgb(
                flat_rgba[index, :3], level, strength, white_threshold)
    return rgba


def paint_point(canvas_rh, canvas_rv, coverage, result, origin, spacing,
                variance_canvas_rh=None, variance_canvas_rv=None, distance_canvas=None):
    """Write one checkpoint's ensemble-mean fields into the world canvas.

    With ``distance_canvas`` a cell is only written when its lateral distance to
    this window's bit column does not exceed the distance of the window that
    painted it before (ties go to the later window).
    """
    x0, z0 = origin
    dx, dz = spacing
    nx, _, nz = result["dims"]
    ratio_bounds = result.get('rv_ratio_bounds', (1., 5.))
    rv_cap_ohmm = result.get('rv_hard_cap_ohmm')
    anisotropy = result.get('anisotropy', 'probit')
    posterior_params = result.get('posterior_ensemble_params', result['posterior_params'])
    include_variance = variance_canvas_rh is not None and variance_canvas_rv is not None
    if include_variance:
        rh, rv, variance_rh, variance_rv, is_2d = posterior_moments(
            posterior_params, result["dims"], ratio_bounds, rv_cap_ohmm, anisotropy)
    else:
        members = [member_fields(params, result["dims"], ratio_bounds, rv_cap_ohmm, anisotropy)
                   for params in posterior_params]
        rh = np.mean([member[0] for member in members], axis=0)
        rv = np.mean([member[1] for member in members], axis=0)
        is_2d = members[0][2]
    ix = int(round((result["shift_ft"]["x"] - x0) / dx))
    iz = int(round((result["shift_ft"]["z"] - z0) / dz))
    lateral = np.abs(np.arange(nx) - nx // 2).astype(float)
    if not is_2d:
        ix += nx // 2
        nx = 1
        lateral = np.zeros(1)
    region = (slice(ix, ix + nx), slice(iz, iz + nz))
    write = np.ones((nx, nz), dtype=bool)
    if distance_canvas is not None:
        distance = np.repeat(lateral[:, None], nz, axis=1)
        write = distance <= distance_canvas[region]
        distance_canvas[region] = np.where(write, distance, distance_canvas[region])
    canvas_rh[region] = np.where(write, rh.reshape(nx, nz), canvas_rh[region])
    canvas_rv[region] = np.where(write, rv.reshape(nx, nz), canvas_rv[region])
    coverage[region] += 1
    if include_variance:
        variance_canvas_rh[region] = np.where(write, variance_rh.reshape(nx, nz), variance_canvas_rh[region])
        variance_canvas_rv[region] = np.where(write, variance_rv.reshape(nx, nz), variance_canvas_rv[region])


def build_curtain(checkpoints, include_variance=False, paint='last'):
    if paint not in ('last', 'nearest-bit'):
        raise ValueError(f'Unknown paint mode {paint!r}')
    spacing = (checkpoints[0][1]["spacing_ft"][0], checkpoints[0][1]["spacing_ft"][2])
    xs, zs, xe, ze = [], [], [], []
    for _, result in checkpoints:
        nx, _, nz = result["dims"]
        xs.append(result["shift_ft"]["x"])
        zs.append(result["shift_ft"]["z"])
        xe.append(result["shift_ft"]["x"] + nx * spacing[0])
        ze.append(result["shift_ft"]["z"] + nz * spacing[1])
    origin = (min(xs), min(zs))
    n_x = int(round((max(xe) - origin[0]) / spacing[0]))
    n_z = int(round((max(ze) - origin[1]) / spacing[1]))
    canvas_rh = np.full((n_x, n_z), np.nan)
    canvas_rv = np.full((n_x, n_z), np.nan)
    coverage = np.zeros((n_x, n_z))
    variance_rh = np.full((n_x, n_z), np.nan) if include_variance else None
    variance_rv = np.full((n_x, n_z), np.nan) if include_variance else None
    distance = np.full((n_x, n_z), np.inf) if paint == 'nearest-bit' else None
    for _, result in checkpoints:
        paint_point(canvas_rh, canvas_rv, coverage, result, origin, spacing,
                    variance_rh, variance_rv, distance)
    if include_variance:
        return canvas_rh, canvas_rv, coverage, origin, spacing, variance_rh, variance_rv
    return canvas_rh, canvas_rv, coverage, origin, spacing


def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float).ravel()
    edges = np.empty(centers.size + 1)
    edges[1:-1] = 0.5 * (centers[1:] + centers[:-1])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def load_reference(benchmark, extent_ft):
    with h5py.File(Path(benchmark) / "globalmodel.h5", "r") as handle:
        rh = handle["formation/Rh"][:, handle["formation/Rh"].shape[1] // 2, :]
        x_ft = handle["formation/x"][:, 0] * meter_to_feet
        z_ft = handle["formation/z"][:, 0] * meter_to_feet
        wx_ft = handle["wellpath/X"][:, 0] * meter_to_feet
        wz_ft = handle["wellpath/Z"][:, 0] * meter_to_feet
    x0, x1, z0, z1 = extent_ft
    keep_x = (x_ft >= x0) & (x_ft <= x1)
    keep_z = (z_ft >= z0) & (z_ft <= z1)
    return {"rh": rh[np.ix_(keep_z, keep_x)],
            "x_edges": centers_to_edges(x_ft[keep_x]),
            "z_edges": centers_to_edges(z_ft[keep_z]),
            "wx": wx_ft, "wz": wz_ft}


def render(checkpoints, args):
    uncertainty_enabled = getattr(args, 'uncertainty_lightness', False)
    paint = getattr(args, 'paint', 'last')
    if uncertainty_enabled:
        (canvas_rh, canvas_rv, coverage, origin, spacing,
         variance_rh, variance_rv) = build_curtain(checkpoints, include_variance=True, paint=paint)
        finite_variances = np.concatenate([
            variance_rh[np.isfinite(variance_rh)],
            variance_rv[np.isfinite(variance_rv)],
        ])
        variance_scale = (float(np.percentile(finite_variances, args.uncertainty_percentile))
                          if finite_variances.size else 0.)
    else:
        canvas_rh, canvas_rv, coverage, origin, spacing = build_curtain(checkpoints, paint=paint)
        variance_rh = variance_rv = None
        variance_scale = 0.
    n_x, n_z = coverage.shape
    x_edges = origin[0] + np.arange(n_x + 1) * spacing[0]
    z_edges = origin[1] + np.arange(n_z + 1) * spacing[1]
    reference = load_reference(args.benchmark, (x_edges[0], x_edges[-1], z_edges[0], z_edges[-1])) \
        if args.benchmark else None
    well_x = np.array([result["shift_ft"]["x"] + result["dims"][0] / 2 * spacing[0]
                       for _, result in checkpoints])
    well_z = np.array([result["shift_ft"]["z"] + result["dims"][2] / 2 * spacing[1]
                       for _, result in checkpoints])
    fidelities = [fidelity_of(result) for _, result in checkpoints]

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[3.2, 1], hspace=0.25, wspace=0.12)
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    misfit_axis = fig.add_subplot(grid[1, :])

    true_field = reference["rh"] if reference else None
    scale = np.log10(true_field) if true_field is not None else np.log10(np.exp(canvas_rh))
    vmin, vmax = np.nanpercentile(scale, [2, 98])
    if getattr(args, 'log10_limits', None) is not None:
        vmin, vmax = args.log10_limits
    cmap = value_colormap(getattr(args, 'colormap', 'viridis'))
    shared = dict(cmap=cmap, vmin=vmin, vmax=vmax)
    value_norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    panels = [("Reference log10 Rh", np.log10(true_field), reference["x_edges"], reference["z_edges"]) if reference
              else ("Reference log10 Rh (unavailable)", None, x_edges, z_edges),
               (f"{getattr(args, 'estimate_label', 'Estimated')} log10 Rh", np.log10(np.exp(canvas_rh)).T, x_edges, z_edges),
               (f"{getattr(args, 'estimate_label', 'Estimated')} log10 Rv", np.log10(np.exp(canvas_rv)).T, x_edges, z_edges)]
    for panel_index, (axis, (title, field, xedge, zedge)) in enumerate(zip(axes, panels)):
        if field is None:
            axis.text(.5, .5, "no benchmark", ha="center", va="center", transform=axis.transAxes)
        elif uncertainty_enabled and panel_index in (1, 2):
            variance = (variance_rh if panel_index == 1 else variance_rv).T
            rgba = uncertainty_rgba(field, variance, cmap, value_norm, variance_scale,
                                    args.uncertainty_strength,
                                    args.uncertainty_white_threshold)
            axis.imshow(rgba, extent=(xedge[0], xedge[-1], zedge[0], zedge[-1]),
                        origin="lower", interpolation="nearest", aspect="auto")
            value_map = ScalarMappable(norm=value_norm, cmap=cmap)
            value_map.set_array([])
            fig.colorbar(value_map, ax=axis, fraction=0.046, pad=0.02)
        else:
            mesh = axis.pcolormesh(xedge, zedge, field, shading="auto", **shared)
            fig.colorbar(mesh, ax=axis, fraction=0.046, pad=0.02)
        for fidelity, color, marker in (("2D", "tab:orange", "o"), ("1D", "tab:blue", "s")):
            pick = np.array([chosen == fidelity for chosen in fidelities])
            if pick.any():
                axis.scatter(well_x[pick], well_z[pick], c=color, marker=marker, s=18,
                             zorder=3, label=f"{fidelity} points ({int(pick.sum())})")
        axis.plot(well_x, well_z, color="white", lw=1.0, zorder=2)
        axis.scatter(well_x[-1:], well_z[-1:], marker="x", color="red", s=60, zorder=4, label="latest")
        axis.set_title(title)
        axis.set_xlabel("X [ft]")
        axis.set_aspect("equal", adjustable="box")
        axis.invert_yaxis()
        if getattr(args, 'shared_extents', False):
            axis.set_xlim(x_edges[0], x_edges[-1])
            axis.set_ylim(z_edges[-1], z_edges[0])
    axes[0].set_ylabel("TVD [ft]")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=.7)
    if uncertainty_enabled:
        mid_color = cmap(value_norm((vmin + vmax) / 2.))[:3]
        levels = (0., .5, 1.)
        labels = ("low variance", "mid variance", "high variance / lighter")
        handles = [Patch(facecolor=hls_lightness_rgb(
            mid_color, level, args.uncertainty_strength,
            args.uncertainty_white_threshold), edgecolor="none", label=label)
                   for level, label in zip(levels, labels)]
        fig.legend(handles=handles, title=(f"Relative posterior variance (p{args.uncertainty_percentile:g} "
                                           f"recomputed per render; >{args.uncertainty_white_threshold:g} white)"),
                   loc="upper center", bbox_to_anchor=(.5, .95), ncol=3,
                   fontsize=8, title_fontsize=8, frameon=False)

    draw_misfit(misfit_axis, checkpoints)
    counts = {fidelity: fidelities.count(fidelity) for fidelity in ("1D", "2D")}
    painted = ' - at-drilling (nearest-bit) estimate' if paint == 'nearest-bit' else ''
    fig.suptitle(f"Inversion curtain - {len(checkpoints)} points "
                 f"(1D: {counts['1D']}, 2D: {counts['2D']}) - {args.run_dir.name}{painted}", y=0.98)
    fig.subplots_adjust(top=0.87 if uncertainty_enabled else 0.92)

    output = args.output or (args.run_dir / "curtain.png")
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    return output


def draw_misfit(misfit_axis, checkpoints, position=None, position_label="Logging point index"):
    """Per-point gate misfit and posterior misfit (1D blue, 2D orange).

    Plotted against the logging index, or against ``position[index]`` (e.g. bit x) when given.
    """
    indices, before, after, thresholds, kinds = [], [], [], [], []
    fidelity_by_index = {index: fidelity_of(result) for index, result in checkpoints}
    for index, result in checkpoints:
        misfit = result.get("misfit") or {}
        posterior = result.get("posterior_misfit")
        if misfit.get("before_final") is not None or posterior:
            indices.append(index)
            before.append(result.get('best_member_data_misfit', misfit.get("before_final")))
            after.append(float(np.mean(posterior)) if posterior else None)
            thresholds.append(result.get("misfit_threshold"))
            kinds.append(prediction_kind(result))
    if indices:
        before = np.array([np.nan if value is None else value for value in before])
        after = np.array([np.nan if value is None else value for value in after])
        fidelity_by_index = {(position[index] if position else index): fidelity
                             for index, fidelity in fidelity_by_index.items()}
        if position is not None:
            indices = [position[index] for index in indices]
        misfit_axis.semilogy(indices, before, "o", ms=4, color="0.4", label="pre-final gate misfit")
        valid = np.isfinite(after)
        for kind, marker, label in [('forward', 'o', 'posterior: forward evaluated'),
                                     ('linearized', 'x', 'posterior: linearized estimate')]:
            pick = valid & (np.asarray(kinds) == kind)
            if pick.any():
                colors = np.where([fidelity_by_index[index] == '2D' for index in np.array(indices)[pick]],
                                  'tab:orange', 'tab:blue')
                misfit_axis.scatter(np.array(indices)[pick], after[pick], c=colors, marker=marker,
                                    s=18, zorder=3, label=label)
        threshold = next((value for value in thresholds if value is not None), None)
        if threshold is not None:
            misfit_axis.axhline(threshold, color="red", ls="--", lw=1, label=f"threshold {threshold:g}")
        misfit_axis.legend(fontsize=8, loc="best")
    misfit_axis.set_xlabel(position_label)
    misfit_axis.set_ylabel("Reduced misfit")
    misfit_axis.grid(alpha=.25)


def main(argv=None):
    args = parse_args(argv)
    if not args.run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
    if not args.follow:
        checkpoints = load_checkpoints(args.run_dir)
        if not checkpoints:
            raise RuntimeError("No complete checkpoints found yet")
        print(f"Rendered {len(checkpoints)} points -> {render(checkpoints, args)}")
        return 0
    signature, cache = None, {}
    while True:
        checkpoints = load_checkpoints(args.run_dir, cache)
        current = tuple(index for index, _ in checkpoints)
        if checkpoints and current != signature:
            signature = current
            print(f"{time.strftime('%H:%M:%S')} rendered {len(checkpoints)} points -> {render(checkpoints, args)}",
                  flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
