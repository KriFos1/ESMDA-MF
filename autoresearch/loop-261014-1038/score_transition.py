"""Score saved transition-zone inversions against Benchmark-3 truth."""

import argparse
import pickle
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BENCHMARK = ROOT.parent / "Dashboard/benchmarks/Benchmark-3/globalmodel.h5"
BASELINE_30 = ROOT / "runs/benchmark3-lm-laplace-t75-lx10-5starts-20261013"
BASELINE_31_35 = ROOT / "runs/benchmark3-lm-laplace-t75-lx10-resume31-5starts-20261013"
INDICES = tuple(range(30, 36))
MIN_RATIO, MAX_RATIO, RV_CAP = 1.0, 3.0, 80.0


def overlap_weights(centers, edges, cell_width):
    low = centers - cell_width / 2
    high = centers + cell_width / 2
    weights = np.empty((len(edges) - 1, len(centers)))
    for i in range(len(weights)):
        span = np.maximum(0.0, np.minimum(edges[i + 1], high) - np.maximum(edges[i], low))
        weights[i] = span / (high - low)
    totals = weights.sum(axis=1, keepdims=True)
    for i in np.flatnonzero(totals[:, 0] <= 0):
        center = (edges[i] + edges[i + 1]) / 2
        weights[i, np.argmin(np.abs(centers - center))] = 1.0
    return weights / weights.sum(axis=1, keepdims=True)


def read_checkpoint(path, truth, x_centers, z_centers, dx_truth, dz_truth):
    result = pickle.loads(path.read_bytes())
    nx, _, nz = result["dims"]
    cells = nx * nz
    model = np.asarray(result["map_model"], dtype=float)
    if model.shape != (2 * cells,) or not np.isfinite(model).all():
        raise ValueError(f"{path}: missing or invalid MAP field")
    dx, dz = result["spacing_ft"][0], result["spacing_ft"][2]
    shift = result["shift_ft"]
    x_edges = shift["x"] + np.arange(nx + 1) * dx
    z_edges = shift["z"] + np.arange(nz + 1) * dz
    wx = overlap_weights(x_centers, x_edges, dx_truth)
    wz = overlap_weights(z_centers, z_edges, dz_truth)
    targets = {name: (wz @ (wx @ field.T).T).T for name, field in truth.items()}

    log_rh = model[:cells].reshape(nx, nz)
    latent = model[cells:].reshape(nx, nz)
    log_rv = np.minimum(log_rh + np.log(MIN_RATIO)
                        + np.log(MAX_RATIO / MIN_RATIO) * ndtr(latent), np.log(RV_CAP))
    errors = {
        "rh": (log_rh - np.log(targets["Rh"])) / np.log(10.0),
        "rv": (log_rv - np.log(targets["Rv"])) / np.log(10.0),
    }
    rmses = {name: float(np.sqrt(np.mean(error ** 2))) for name, error in errors.items()}
    from esmda_core import measurement_misfit, noise_floor_mask, observation_vectors
    from run_DA_sequential import component_positions

    data = pd.read_pickle(ROOT / "runs/benchmark3-data-20261002/data.pkl")
    variances = pd.read_pickle(ROOT / "runs/benchmark3-data-20261002/var.pkl")
    observations, covariance = observation_vectors(
        data, variances, result["logging_index"], result["tools"])
    sub_cell = np.zeros(len(observations), dtype=bool)
    sub_cell[component_positions(len(observations))] = True
    fixed_mask = ~(noise_floor_mask(observations, covariance) | sub_cell)
    saved_mask = np.asarray(result["observation_mask"], dtype=bool)
    if not np.array_equal(fixed_mask, saved_mask):
        raise ValueError(f"{path}: variance treatment changed the scored channel mask")
    prediction = np.asarray(result["map_prediction"], dtype=float)
    misfit = measurement_misfit(prediction, observations[fixed_mask], covariance[fixed_mask])
    if not np.isfinite(misfit) or not np.isfinite(list(rmses.values())).all():
        raise ValueError(f"{path}: non-finite score component")
    return {"misfit": misfit, **{f"rmse_{key}": value for key, value in rmses.items()},
            "fidelity": result["selected_fidelity"],
            "converged": bool(result["map_converged"])}


def rows_for(run_dir, truth, x_centers, z_centers, dx_truth, dz_truth,
             allow_missing=False):
    rows = {}
    for index in INDICES:
        if index == 30:
            candidate = run_dir / "inversion_results_assim_30.pkl"
            path = candidate if candidate.exists() else BASELINE_30 / candidate.name
        else:
            path = run_dir / f"inversion_results_assim_{index}.pkl"
        if not path.exists():
            if allow_missing:
                rows[index] = None
                continue
            raise FileNotFoundError(f"Missing index {index}: {path}")
        rows[index] = read_checkpoint(path, truth, x_centers, z_centers,
                                      dx_truth, dz_truth)
    return rows


def aggregate(rows):
    mean_misfit = float(np.mean([rows[i]["misfit"] for i in INDICES]))
    field_rmse = float(np.sqrt(np.mean([
        (rows[i]["rmse_rh"] ** 2 + rows[i]["rmse_rv"] ** 2) / 2
        for i in INDICES
    ])))
    return mean_misfit, field_rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    args = parser.parse_args()

    with h5py.File(args.benchmark, "r") as handle:
        x_centers = handle["formation/x"][:, 0] / 0.3048
        z_centers = handle["formation/z"][:, 0] / 0.3048
        mid = handle["formation/Rh"].shape[1] // 2
        truth = {name: np.asarray(handle[f"formation/{name}"][:, mid, :], dtype=float)
                 for name in ("Rh", "Rv")}
    dx_truth = float(np.median(np.diff(x_centers)))
    dz_truth = float(np.median(np.diff(z_centers)))

    baseline_rows = rows_for(BASELINE_31_35, truth, x_centers, z_centers,
                             dx_truth, dz_truth)
    candidate_rows = rows_for(args.run_dir.resolve(), truth, x_centers, z_centers,
                              dx_truth, dz_truth, allow_missing=True)
    base_misfit, base_rmse = aggregate(baseline_rows)
    failures = sum(row is None or not row["converged"] for row in candidate_rows.values())
    if failures:
        score = 1e6 + failures
    else:
        candidate_misfit, candidate_rmse = aggregate(candidate_rows)
        score = float(np.sqrt((candidate_misfit / base_misfit)
                              * (candidate_rmse / base_rmse)))
    print(f"{score:.12g}")


if __name__ == "__main__":
    main()
