# Changes to the 2D simulator (fwd2D) and how the inversion uses it

The gated EnIF-MDA chain uses the 2D EM simulator (LogSimulator3D v4.2, `fwd2D`) only
in its MAP/Laplace stage. That stage needs exact derivatives of the 8 UDAR components
with respect to ln Rh and ln Rv on the inversion window. The upstream code cannot supply
them, so the production binaries are patched builds. This note lists every change to the
2D source and its builds: what was changed, why, where it lives, how it was checked, and
what is still open. The last section proposes tests.

The patches are applied by `Dashboard/scripts/build_discrete_adjoint.py`, together with
`Dashboard/code/solver-adjoint/adjoint_capture.f90`. The script copies `src/*.f90` of a
source tree into a fresh directory, patches only those copies, and compiles with
`mpiifx -O3 -qopenmp -qmkl`. The source trees themselves are never modified.

## Build lineage

| Build | Location (HPC unless noted) | Content | Use |
|---|---|---|---|
| Original v4.2 | `/cluster/home/krfo/LogSimulator3D_v4.2fullfortran/` | upstream (Oct 2024) | reference only; rejects datflag > 4 |
| jacobi-nrx | `/cluster/home/krfo/LogSimulator3D_v4.2fullfortran_jacobi_nrx_2D_3D/` | upstream variant: in-solver Jacobian, datflag 5 | **source tree of every build below**; generated the Benchmark-2/3 data |
| adjoint-2d-v1 | `/cluster/home/krfo/adjoint-2d-v1/` | §2 and §3 | gradient-mode MAP stages (older runs) |
| adjoint-2d-v2 | `~/adjoint-2d-v2` | v1 + OpenMP pragmas in `adjoint_capture.f90` | never used in production |
| adjoint-2d-v3 | Jupiter1 `~/adjoint-2d-v3` | v1 + block-ROM safety patch (§5) | **not deployed** |
| adjoint-2d-v1-orient | `/cluster/home/krfo/adjoint-2d-v1-orient/` (Jupiter1: `/data1/krfo/esmda-node/adjoint-2d-v1-orient`) | v1 + `orientation.inp` (§4) | **production** (bedding mode) |

## 1. jacobi-nrx compared with the original distribution (upstream, not our patch)

The jacobi-nrx tree carries the same version tag as the original but differs from it:

- **New files:** `jacobianprocessing.f90` and `write_read_h5.f90`.
- **`input.f90`:**
  - `outputsettings.inp` gains line 3 (`jacobiflag`) and line 4 (`jacthreshold`).
  - The datflag range check is commented out.
- **`fwd2D*.f90`:** `jacobiflag = 1` routes to `bai_em_jacobi`.
- **`buildgrid.f90`:** adds `f_rxpoint`, `jacsign` and `rhL/rvL`.
- **`bai_em_matrix_mod2D.f90`:**
  - `SQTENS` perturbs σh/σv by +10 %.
  - Adds `getLGrhrv` and `computeJacSparse`.
- **`signalconversion.f90` / `output.f90`:**
  - datflag 5 (16 UDAR3D components).
  - `dat.dat` is written to `outputs/`.

**Validation.** On the native Benchmark-3 model, the original, jacobi-nrx, v1 and orient (gradient mode)
all reproduce `outputs_2D` at χ²/n 0.000 (indices 10, 33, 40, 44) and 0.0004 (index 130).
Source: `MULTISTART_IDX130_20261023.md`, run `runs/native-model-check-20261023`.

**Caveat: the in-solver Jacobian of this tree is unusable.** In a matched original-vs-original
finite-difference test, 0 of 15 directions passed, with a best scale ≈ 0 and cosines between −0.61 and
0.15 (`JACOBIAN_ORIGINAL_MATCHED_INDEX45_20261019.md`). Reasons:

- It uses separate +10 % perturbations and an Rh/Rv rescale.
- It interpolates nodal sensitivities onto cells.

## 2. Discrete-adjoint instrumentation (`build_discrete_adjoint.py`, v1 and later)

Three source files differ from jacobi-nrx:

1. **Material capture.** In `SUBROUTINE SQTENS` of `bai_em_matrix_mod2D.f90`:
   - Adds `use adjoint_capture` and calls `begin_material(...)` first.
   - After each of the four staggered `SQSIG(...,k)=SQS(:)*SQAMU` assignments, calls
     `record_material(...)`. It records σh, σv, the rotation cos/sin, whether `gradan` was used,
     and the gradient node.
   - Ends with `save_material(f_rxpoint)`.
   - Tapes: `outputs/adjoint/material_rx<n>.bin` and `grid_rx<n>.txt`.
   - The build parses the eight averaging weights and the denominator of every stencil into
     `adjoint-spec.json`, together with the SHA-256 of the unpatched source.
2. **Kernel export.**
   - `computeJacSparse` is replaced by a stub that calls `export_discrete_kernel`.
   - That routine writes ∂(8 UDAR components)/∂(6 conductivity-tensor entries × 4 stencils) per node
     to `outputs/adjoint/kernel_tool<n>.bin`.
   - It stops unless datflag = 3.
3. **No perturbed assemblies.** In `bai_em_main_mod.f90`:
   - `CALL SQTENS(logpoint,sqsigh/sqsigv)` becomes `sqsigh=sqsig` / `sqsigv=sqsig`.
   - The `process_jacobian` call is removed.
   - The model-space chain rule runs in Python instead: `pxwf.execution.adjoint.finish`, which
     writes `outputs/jacobian/jac_logpt1_tool<n>.h5['jac']` with shape `(2, 8, nz, ny, nx)`. The
     pullback covers the tensor rotation, arithmetic/harmonic averaging, the SVD-orientation
     derivative, and the nearest-cell sampling.

**Validation:**
- 8 of 8 directional finite-difference (FD) checks passed at 2 % (errors 0.1–1.2 %;
  `HPC_TESTING.md`, `runs/hpc-2d-validation-final-20261002`).
- On a planar control model: 0.13–0.50 % with O(h²) convergence (`JACOBIAN_ORIENTATION_AND_CONTROL_20261019.md`).
- The failures at index 45 are forward-map nonlinearity, not adjoint error (Spearman −0.72
  between the linearity measure and the FD error).
- Removing the orientation term gives 63–108 % errors, so that term is required.

## 3. Phase-derivative fix (`phase_source`, `jacobianprocessing.f90`)

`calcPha` (four factors) and `calcPha2` (two factors) are replaced by the exact derivative of
−180/π · atan2 of the product/quotient:

    V = -180/pi * sum(± (a_i db_i - b_i da_i) / (a_i^2 + b_i^2))

The sign is + for numerator factors and − for denominator factors. The legacy formulas did not
match the forward's atan2 definitions. USDP's opposite sign stays in `getdgeosig` (`V(1)=-V(1)`).
The fix is in all adjoint builds and is covered by the FD checks of §2. Phases are wrapped in the
differences there.

## 4. Selectable material orientation (`orientation.inp`, `buildgrid.f90`, v1-orient)

**The problem.** Upstream hard-codes `f_ifdip = -1`: every inner cell takes its TI axis from
the dominant gradient of −2 ln Rh − ln Rv, however small that gradient is. On smooth inversion
models this flips axes between neighbouring cells:

- an 89° flip in one cell gave a 7.8σ jump at index 33;
- the Jacobian shows spikes of 10⁶–10¹⁰.

**The patch** (`orientation_source`):
- Directly after `f_ifdip=-1`, an optional `orientation.inp` is read; its first integer
  overrides `f_ifdip` (0 = the bedding/well-dip axis everywhere).
- The value used is printed as `buildgrid: material orientation ifdip = <v>`.
- Without the file the binary behaves exactly as before.
- The spec moves to version 2 and gains `orientation_input`.

**How the worker uses it** (pxwf `worker.py`):
- `orientation: bedding` writes `orientation.inp` = 0.
- Builds without the capability are refused.
- After every solve, the worker checks that each rank logged the requested `ifdip`.
- In ESMDA-MF the setting is selected with `--orientation` or `DA_SIM_ORIENTATION`.

**Validation:**
- Gradient mode is bit-identical to v1 (14 solves).
- Bedding mode:
  - Jacobian cosines 0.957–1.000;
  - the response per unit step is constant across step sizes;
  - 0 of 4048 edges use the gradient axis.

**Consequence: the data had to be regenerated.** Bedding is a forward-model change. The original
gradient-mode Benchmark-3 data are inconsistent with it (χ²/n up to 58 on the native model).
`outputs_2D` was therefore regenerated in bedding mode on 2026-10-23:

- tolerance 1e-6, single-point runs (index 12 at 1e-5);
- the original is kept in `outputs_2D_gradient_20260918/`;
- details in `Dashboard/benchmarks/Benchmark-3/OUTPUTS_2D_PROVENANCE.md`.

**All current chain runs use bedding mode with the regenerated data.**

## 5. Block-ROM crash: diagnosis and partial patch (v3, **not deployed**)

**The crash.** `block_rom_` segfaults (exit 174) on thick, deep, resistive sheets:

- The Gauss–Radau bound goes NaN.
- A progress-print `repeat(...)` with a garbage count then corrupts the heap.
- Whether it happens depends on the thread count.

**The patch.** `patch_blockrom.py` patches a build copy of `bai_em_atuzi_mod.f90`:

- **P1:** clamp the repeat counts and guard non-finite bounds.
- **P2:** a non-finite bound becomes a clean breakdown with a `convergence_error.txt` receipt.
- **P3:** add INFO checks after `DPOSV/dgetrf/dgetri`.

**Still open:**
- One case still segfaults. Suspect: in `blocktridiagonalsolver`, the 14 → 13-row assignment
  to `mtrQR`.
- A `-check all` audit is blocked by an `svd` shape mismatch in `buildgrid.f90`.

Production does **not** include this patch. The ESMDA-MF adapter recognizes exit 174 from the
batch receipts (`classify_solver_start_failure` / `is_solver_exit_174`), so callers can treat it
as a failed evaluation; in the LM a failed candidate is simply not accepted.

## 6. Custom-grid file name (left in place on purpose)

- `input.f90` reads `customgrids.inp`.
- The pxwf worker writes `customgrid.inp`, so the built-in grid is always used (uniform 20 4 40,
  optimal 12 10 16, averaging 2 1 2).
- The benchmark data were generated with that built-in grid. Fixing the name would break
  agreement (native-model χ²/n 0.002–70 with the distribution's grid, compared with 0.000).
- As a result, `--solver-grid` / `DA_SIM_SOLVER_GRID` have no effect.

## 7. Interface and runtime facts the inversion relies on

**Inputs the worker writes:**
- `globalmodel.h5`: the 2D tile repeated over `y_layers`; Inclination and Azimuth all zero.
- `selection.inp` = `1/1/0/tol`: one logging point, depthref 0 (tool midpoint).
- `outputsettings.inp` = `3/1/<jac>/1e-10`: datflag 3, i.e. 8 components.
- `orientation.inp` (optional).

**Tolerance:**
- The chain and the regenerated data use 1e-6.
- Tolerances of 1e-7 and 1e-8 change predictions by ≤ 0.0012σ.

**MKL and threading:**
- MKL rounding moves predictions by 0.4–1.1σ between thread and CBWR settings.
  Every configuration is deterministic.
- `SimBackendConfig` fixes `mkl_threads`, `mkl_cbwr`, `srun_steps` and `physical_cores` per run;
  they are part of the forward identity recorded in `study.json`.
- Only two MPI ranks help.
- The recommended layout is 2 ranks × 8 cores with `MKL_CBWR=AVX2,STRICT`: about 2.7 s per
  forward run and 7.5–8 s with the Jacobian.

**Other limits:**
- The 2D adjoint supports datflag 3 only. The 3D stack (`../ESMDA-3D/build/adjoint_capture.f90`)
  adds datflag 5.
- Single-point runs are used throughout: a multi-point run hung in its final MPI step.

## 8. Python-side changes in this branch that depend on the solver changes

**`esmda_remote2d.py`:**
- `SimBackendConfig.orientation` (`--orientation`, `DA_SIM_ORIENTATION`).
- `mkl_threads/mkl_cbwr/srun_steps/physical_cores` (exact `srun` steps inside an allocation).
- The node backend (SSH host without Slurm).
- Solver-failure classification (`classify_solver_start_failure`, `is_solver_exit_174`).
- Environment round-trip (`to_env`).

**`esmda_core.py`:**
- `align_phases`: data-space phase wrapping, separate from the derivative fix of §3.
- Noise-floor mask, anisotropy parameterizations (`log_clipped`) and their Jacobian chain rule
  (`optimizer_jacobian`).

**`pilot_rml_index45.py`:**
- Requires `adjoint-spec.json` next to `fwd2D` (`trusts_ad('2D')`).
- Composes the window Jacobian.
- Supports graded solver rows (`grid_maps`).

## 9. Suggested tests

The existing Python tests cover the adapters (`tests/test_sequential.py`, `test_lm_laplace.py`,
`test_pilot_rml_index45.py`, `test_graded_grid.py`). The solver changes themselves are only
checked by one-off validation runs. Suggested additions:

**Build and patch tests (run on HPC or Jupiter1):**
1. **Patcher regression.** Run each of `material_source`, `orientation_source` and
   `phase_source` on the pinned jacobi-nrx sources. Assert the resulting
   `adjoint-spec.json` (stencil count 4, 8 samples each, `source_sha256`) and a golden diff
   of the patched files. A changed upstream file must fail loudly.
2. **Gradient-mode identity.** v1-orient without `orientation.inp`, and with `ifdip=-1`, must
   reproduce v1 bit for bit on 3 fixed window models.
3. **Native-data reproduction.** On the native Benchmark-3 model, bedding mode at tolerance 1e-6
   must reproduce `outputs_2D` (χ²/n < 1e-3) at a fixed set of indices (e.g. 10, 33, 40, 44, 130).
   Gradient mode must reproduce `outputs_2D_gradient_20260918/`.

**Jacobian tests:**
4. **Adjoint vs central FD, smooth model.**
   - Planar, smooth model; ln Rh and ln Rv; one cell and smooth directions.
   - Steps h ∈ {0.01, 0.003}, both orientations.
   - Thresholds: relative error < 1 %, O(h²) convergence.
5. **Phase derivative.** Same setup, restricted to the phase channels (USDP/UADP/UHRP/UHAP),
   near the ±180° cut, to pin the §3 sign convention.
6. **Orientation term.** In gradient mode, check that dropping `orientation_pullback` breaks
   test 4 (guards the term). In bedding mode, check that the material tapes report 0 gradient-axis edges.

**Robustness and interface tests:**
7. **Datflag guard.** A datflag-5 request to a 2D adjoint build must fail with the
   `error stop` message, not produce silent output.
8. **Block-ROM guard (when v3 is deployed).** The three crash fields from
   `SOLVER_BLOCK_ROM_DEBUG.md` must end with a convergence-error receipt, not exit 174. 20
   ordinary fields must stay within 0.1σ of v1.
9. **Custom-grid pin.** Assert that the worker writes `customgrid.inp` and that the solver log
   shows the built-in 44×24×92 grid. This test exists to fail if someone "fixes" the name
   without regenerating the data.
10. **Determinism.** Two runs with identical `SimBackendConfig` (threads, MKL CBWR) must give
    bit-identical predictions and Jacobians. A different MKL configuration may differ, but by
    < 1.5σ.
