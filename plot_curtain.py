"""Gradually build the inversion curtain from sequential checkpoints.

Reads ``inversion_results_assim_<index>.pkl`` files from a run directory,
places every local-window posterior at its world position, and renders the
estimated Rh/Rv curtains next to the benchmark reference, with the well
trajectory and the per-point measurement misfit. ``--follow`` re-renders
whenever new checkpoints appear, so the curtain grows during a run.
"""

import argparse
import pickle
import re
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from run_DA_sequential import max_ratio, meter_to_feet, min_ratio

CHECKPOINT_RE = re.compile(r"inversion_results_assim_(\d+)\.pkl$")


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Sequential run output directory")
    parser.add_argument("--benchmark", type=Path, default=here.parent / "Dashboard/benchmarks/Benchmark-3",
                        help="Benchmark root for the reference curtain (blank string disables)")
    parser.add_argument("--output", type=Path, default=None, help="Output image (default <run-dir>/curtain.png)")
    parser.add_argument("--margin-ft", type=float, default=50., help="Curtain margin around the well")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--follow", action="store_true", help="Keep re-rendering as checkpoints appear")
    parser.add_argument("--interval", type=float, default=60., help="Follow-mode poll interval, seconds")
    args = parser.parse_args(argv)
    if args.dpi < 20 or args.interval <= 0 or args.margin_ft < 0:
        parser.error("Require dpi >= 20, interval > 0 and margin >= 0")
    return args


def load_checkpoints(run_dir):
    checkpoints = []
    for path in sorted(run_dir.glob("inversion_results_assim_*.pkl")):
        match = CHECKPOINT_RE.search(path.name)
        if match is None:
            continue
        try:
            with path.open("rb") as stream:
                result = pickle.load(stream)
        except (pickle.UnpicklingError, EOFError, AttributeError):
            continue
        if not np.isfinite(np.asarray(result["posterior_params"], dtype=float)).all():
            continue
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


def member_fields(params, dims):
    """One member's (ln Rh, ln Rv) on the window; 1D members return columns."""
    nx, _, nz = dims
    params = np.asarray(params, dtype=float)
    n = nx * nz
    if params.size == 2 * n:
        log_rh = params[:n].reshape(nx, nz)
        latent = params[n:].reshape(nx, nz)
        log_ratio = np.log(min_ratio) + np.log(max_ratio / min_ratio) * norm.cdf(latent)
        return log_rh, log_rh + log_ratio, True
    log_rh = params[:nz]
    latent = params[nz:]
    log_ratio = np.log(min_ratio) + np.log(max_ratio / min_ratio) * norm.cdf(latent)
    return log_rh, log_rh + log_ratio, False


def paint_point(canvas_rh, canvas_rv, coverage, result, origin, spacing):
    """Write one checkpoint's ensemble-mean fields into the world canvas."""
    x0, z0 = origin
    dx, dz = spacing
    nx, _, nz = result["dims"]
    members = [member_fields(params, result["dims"]) for params in result["posterior_params"]]
    rh = np.mean([member[0] for member in members], axis=0)
    rv = np.mean([member[1] for member in members], axis=0)
    ix = int(round((result["shift_ft"]["x"] - x0) / dx))
    iz = int(round((result["shift_ft"]["z"] - z0) / dz))
    if not members[0][2]:
        ix += nx // 2
        nx = 1
    canvas_rh[ix:ix + nx, iz:iz + nz] = rh.reshape(nx, nz)
    canvas_rv[ix:ix + nx, iz:iz + nz] = rv.reshape(nx, nz)
    coverage[ix:ix + nx, iz:iz + nz] += 1


def build_curtain(checkpoints):
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
    for _, result in checkpoints:
        paint_point(canvas_rh, canvas_rv, coverage, result, origin, spacing)
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
    canvas_rh, canvas_rv, coverage, origin, spacing = build_curtain(checkpoints)
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
    shared = dict(cmap="viridis", vmin=vmin, vmax=vmax)

    panels = [("Reference log10 Rh", np.log10(true_field), reference["x_edges"], reference["z_edges"]) if reference
              else ("Reference log10 Rh (unavailable)", None, x_edges, z_edges),
              ("Estimated log10 Rh", np.log10(np.exp(canvas_rh)).T, x_edges, z_edges),
              ("Estimated log10 Rv", np.log10(np.exp(canvas_rv)).T, x_edges, z_edges)]
    for axis, (title, field, xedge, zedge) in zip(axes, panels):
        if field is None:
            axis.text(.5, .5, "no benchmark", ha="center", va="center", transform=axis.transAxes)
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
    axes[0].set_ylabel("TVD [ft]")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=.7)

    indices, before, after, thresholds, kinds = [], [], [], [], []
    fidelity_by_index = {index: fidelity_of(result) for index, result in checkpoints}
    for index, result in checkpoints:
        misfit = result.get("misfit") or {}
        posterior = result.get("posterior_misfit")
        if misfit.get("before_final") is not None or posterior:
            indices.append(index)
            before.append(misfit.get("before_final"))
            after.append(float(np.mean(posterior)) if posterior else None)
            thresholds.append(result.get("misfit_threshold"))
            kinds.append(prediction_kind(result))
    if indices:
        before = np.array([np.nan if value is None else value for value in before])
        after = np.array([np.nan if value is None else value for value in after])
        misfit_axis.semilogy(indices, before, "o", ms=4, color="0.4", label="after 1D (gate)")
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
    misfit_axis.set_xlabel("Logging point index")
    misfit_axis.set_ylabel("Reduced misfit")
    misfit_axis.grid(alpha=.25)
    counts = {fidelity: fidelities.count(fidelity) for fidelity in ("1D", "2D")}
    fig.suptitle(f"Inversion curtain - {len(checkpoints)} points "
                 f"(1D: {counts['1D']}, 2D: {counts['2D']}) - {args.run_dir.name}", y=0.98)
    fig.subplots_adjust(top=0.92)

    output = args.output or (args.run_dir / "curtain.png")
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    return output


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
    signature = None
    while True:
        checkpoints = load_checkpoints(args.run_dir)
        current = tuple(index for index, _ in checkpoints)
        if checkpoints and current != signature:
            signature = current
            print(f"{time.strftime('%H:%M:%S')} rendered {len(checkpoints)} points -> {render(checkpoints, args)}",
                  flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
