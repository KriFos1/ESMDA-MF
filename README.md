# ESMDA-MF

Multi-fidelity ES-MDA for EM inversion: local 0D/1D ensemble updates followed
by gradient-based 2D inversion on a small moving window. The 2D simulator can
run locally, through Slurm on HPC, or through an SSH node backend.

## Environment

Use the sibling Dashboard project environment, which supplies NumPy, SciPy,
pandas, h5py, EMsim/UTAEM, PET (`pipt`), geostat and `pxwf.execution`:

```bash
../Dashboard/.venv/bin/python run_DA_sequential.py --help
../Dashboard/.venv/bin/python validate_2d.py --help
```

The remote worker requires NumPy and h5py, the `fwd2D` executable, Intel
OneAPI and HDFql. Gradient inversion requires an **exact discrete-adjoint
build with `adjoint-spec.json`**. Runtime identity is recorded in `run.json`
and in each batch request. `PXWF_SRC` can override the sibling Dashboard
source location.

## Local inversion window

The full benchmark grid is the reference/data source. It does **not** set
the inversion-window dimensions.

| Option | Default | Meaning |
|---|---:|---|
| `--nx` | 15 | Local x columns |
| `--nz` | 20 | Local z cells |
| `--y-layers` | 5 | Copies of the 2D field along y |
| `--dx-ft` | 5 | x/y cell width, feet |
| `--dz-ft` | 5 | z cell height, feet |

The default window is **75 × 100 ft**, with 300 independent cells and 600
optimizer parameters. Its dense two-field covariance occupies about 2.75
MiB. The first logging point anchors the central cell; later movements are
snapped to integer x/z cell offsets. Multi-cell and negative movements are
supported. Geometry stays fixed throughout each point's ensemble evaluations
and LM trials. Previous posterior fields are reindexed into the new window,
with edge-cell extension, and conditioned on the current 1D column.

All fidelities use the eight canonical UDAR channels per tool:
`USDP, USDA, UADP, UADA, UHRP, UHRA, UHAP, UHAA`.
Local and remote 2D use the same standalone worker. Full-grid log-resistivity
derivatives are reordered into optimizer cell order, then transformed to
`[ln(Rh), anisotropy latent]`.

## Generate observations

Run from this repository root. The default benchmark is
`../Dashboard/benchmarks/Benchmark-3`.

```bash
../Dashboard/.venv/bin/python write_data.py \
  --output-dir runs/benchmark3-data
```

`write_data.py` discovers all `UDAR3D_*.las` files in `outputs_2D`, selects
channels by header name, and verifies the LAS measured depths against the
benchmark trajectory. It writes `data.pkl`, `var.pkl`, `assim_index.csv` and
`datatyp.csv` to a fresh output directory. Variance cells are rowwise
`['ABS', eight_variances]`, using a default 2.5% noise standard deviation
scaled by mean absolute signal across tools at that logging point.

The inversion reader also accepts the older replicated full-sequence variance
envelopes, selecting the correct logging-point entry.

## Validate the 2D HPC adapter

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
../Dashboard/.venv/bin/python -u validate_2d.py \
  --output-dir runs/hpc-2d-validation \
  --sim-backend slurm --host HPC \
  --partition comp --account energytech --qos normal \
  --nodes 1 --ranks 1 --cpus 5 --memory 16 \
  --walltime 00:20:00 --concurrency 1 \
  --binary-dir /cluster/home/krfo/adjoint-2d-v1 \
  --solver-tolerance 1e-8
```

This seeded heterogeneous 8 × 12 case checks forward/Jacobian prediction
agreement, full-grid shapes, and single-cell/random-direction derivatives for
both optimizer parameter blocks. Default central-difference steps are
`0.003` and `0.001`. Both AD/FD agreement and FD step-to-step agreement must
be within **2% relative L2**. `--steps` accepts alternative steps for a
diagnostic sweep. Very small steps can be precision-limited; larger steps
can leave the local linear regime. The script saves `base.npz` and all checks
in `report.json`, and exits nonzero on failure.

## Short Benchmark-3 HPC pilot

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
../Dashboard/.venv/bin/python -u run_DA_sequential.py \
  --output-dir runs/benchmark3-hpc-smoke \
  --data runs/benchmark3-data/data.pkl \
  --variance runs/benchmark3-data/var.pkl \
  --sim-backend slurm --host HPC \
  --partition comp --account energytech --qos normal \
  --nodes 1 --ranks 1 --cpus 5 --memory 16 \
  --walltime 00:20:00 --concurrency 1 \
  --binary-dir /cluster/home/krfo/adjoint-2d-v1 \
  --start 80 --stop 83 --frequency 1 \
  --ensemble-size 8 --opt-members 1 \
  --max-iterations 1 --max-lm-trials 1 --force-2d
```

Output directories must be fresh. `--stop` is exclusive. `--seed` defaults to
2026. `--force-2d` guarantees the gradient path is exercised even if the
lower-fidelity stages converge. Default full-run sizing is 250 ES-MDA
members, five optimized members, five LM iterations and two trials per
iteration.

### Adaptive fidelity selection

Fidelity escalation is gated by the **measurement-only reduced misfit**

```
chi2_red = mean_i( (f_i - d_i)^2 / Cd_i )
```

evaluated with the ensemble-mean prediction against the unperturbed
observations and the **original** (uninflated) measurement covariance. The
same criterion stops the 0D/1D ES-MDA iteration loops.

- After 1D, if `chi2_red <= --misfit-threshold` (default 4), the remaining
  MDA weight is consumed by **one final 1D ES-MDA update** with inflation
  `1/(1-w)`. No 2D forward or gradient solves are submitted. If that final
  update increases the misfit, the outcome is **reported only**
  (`final_iteration_worsened` flag) — the state is kept, never reverted.
- If `chi2_red > threshold` (or `--force-2d`), the 2D gradient stage runs
  with the same remaining weight `1/(1-w)`.

The total inverse-inflation weight always sums to 1 across 0D, 1D and the
final step. After a 1D-only point, the next 2D prior is built cold from the
1D columns (no stale 2D field is transported).

### Checkpoints

Each `inversion_results_assim_<index>.pkl` records prior/posterior
parameters, predictions, losses, optimizer history, observations, covariance,
geometry, previous logging index, trial errors, and the fidelity-decision
audit: `selected_fidelity`, `misfit` (per-stage reduced misfit, before/after
the final update), `misfit_threshold`, `posterior_misfit`, `two_d_calls`,
`mda_weight`, `lower_fidelity_weight`, `final_inflation_factor` and
`final_iteration_worsened`. `jobs/` holds durable worker requests, results
and batch timings.

## Inversion curtain

```bash
../Dashboard/.venv/bin/python plot_curtain.py --run-dir runs/<name> [--follow]
```

Renders the reference log-Rh curtain, the ensemble-mean estimated Rh/Rv
curtains in world coordinates, the well trajectory (colored by selected
fidelity) and the per-point reduced misfit with the escalation threshold.
`--follow` re-renders `curtain.png` whenever new checkpoints appear
(`--interval`, default 60 s), so the curtain grows during a run. 1D-only
points appear as single-cell columns at the bit.

## Backend options

CLI flags override the corresponding `DA_SIM_*` environment variables.

| Flags | Defaults | Purpose |
|---|---|---|
| `--sim-backend` | `local` | `local`, `slurm`, or `node` |
| `--host`, `--root`, `--python` | `HPC`, `pxwf-runs`, `pxwf-runtime/.venv/bin/python` | Remote target/runtime |
| `--partition`, `--account`, `--qos` | `comp`, blank, blank | Slurm scheduling |
| `--nodes`, `--ranks`, `--cpus`, `--memory` | 1, 1, 8, 16 GiB | Resources **per array element** |
| `--walltime` | `01:00:00` | Per-element time limit |
| `--concurrency`, `--chunk` | 4, 1 | Array concurrency/evaluations per task |
| `--workers` | 8 | SSH-node worker pool |
| `--oneapi`, `--hdfql`, `--binary-dir` | auto-discover | Runtime overrides |
| `--mpi`, `--pmi` | `pmi2`, `/usr/lib64/libpmi2.so` | MPI runtime |
| `--scratch` | shared filesystem | Optional node-local scratch |
| `--solver-tolerance` | `1e-6` | 2D solver tolerance (`DA_SIM_TOLERANCE`) |

Local 2D honors CPU count, concurrency and runtime overrides. Slurm ranks
control MPI parallelism **within one solve**; concurrency controls array
elements, not ranks. Sequential inversion currently submits one member per
batch, so increasing concurrency alone does not parallelize its optimizer.

## Regression checks

```bash
../Dashboard/.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_sequential.py
```

The tests cover import/help isolation, geometry, field/Jacobian layout, the
log-parameter chain rule, rowwise/legacy variances, LAS channel ordering,
conditioning columns, accepted-state bookkeeping, the reduced-misfit
definition, and a three-point controller with both x/z shifts, adaptive
1D/2D gating, the final 1D update, worsened-misfit reporting, cold-start 2D
priors after 1D-only points and lower-fidelity early convergence.

See [HPC_TESTING.md](HPC_TESTING.md) for the verified runtime and recorded
2026-10-02 validation/pilot results.
