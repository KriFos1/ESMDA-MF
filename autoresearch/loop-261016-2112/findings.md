# Indices 25–40: linear-taper ten-step ES-MDA study

**Outcome:** No tested 2D handoff achieves the reference reduced-misfit target
of 75 at indices 31–40. There is no defensible optimum number of 2D steps
within the admissible range 1–5; five steps are insufficient. A threshold
alone cannot fix this 2D model/ensemble mismatch.

## Design and verification

- Independent pointwise replay at indices **25–40**, using each index's saved
  incoming 400-member prior from
  `runs/benchmark3-full-two-step-esmda-b025-t75-20261015-r2`. No candidate is
  chained into the next logging point. The saved index-0-anchored moving window,
  observation mask and original variances are checked at every point.
- Ten likelihood-weight-conserving covariance inflations decrease linearly
  from 15.5234 to 6.2094 (sum of inverse inflations is one). Steps 1–3 use 0D;
  steps 4–5 use 1D; the remaining five slots are either 1D or 1–5 2D steps.
  All 1D and 2D updates use PET sigmoid autoADAloc with correlation midpoint
  **0.3**. The 2D branch ranks all 400 1D members by 1D reduced misfit at its
  handoff and selects the **25 best**. Original-covariance reduced chi-square is
  recomputed from independently evaluated final forward predictions.
- Tested gate thresholds: 25, 50, 75, 100, 150, 250, applied to the shared
  **step-5** ensemble-mean 1D misfit. Each 2D count uses all ten MDA slots;
  the 1D/2D transition moves from after step 9 (one 2D) to after step 5
  (five 2D). Offline thresholds reuse the same completed trajectories.
- 25 Slurm array workers were requested in parallel (comp/energytech,
  5 CPUs/16 GiB per field, pinned exact-adjoint build, solver tolerance 1e-6).
  **5,743** 2D forward calls across the 80 point/step-count branches;
  **976** non-finite 2D evaluations recorded without replacement.
- Benchmark log10 Rh RMSE compares the inversion-window ensemble mean with
  cell-overlap-averaged truth. The 1D branch averages 400 members; 2D branches
  average only solver-valid selected members, so RMSE comparisons are not a
  paired equal-size ensemble estimate when members fail.

## Findings

| Gate | 2D steps | Scored points | Mean Rh RMSE (log10) | Mean reduced misfit | Min 2D survivors |
|---:|---:|---:|---:|---:|---:|
| 50–250 | 0 | 16/16 | 0.64538 | 400.00 | n/a |
| 50–250 | 1 | 15/16 | not defined | not defined | 0 |
| 50–250 | 2 | 16/16 | 0.66822 | 9211.02 | 2 |
| 50–250 | 3 | 16/16 | 0.67892 | 1233.47 | 2 |
| 50–250 | 4 | 15/16 | not defined | not defined | 1 |
| 50–250 | 5 | 15/16 | not defined | not defined | 0 |

The first six indices (25–30) have final 1D misfits of **3.65–27.47** and
need **zero** 2D steps to meet 75. All ten indices 31–40 have final 1D
misfits **130.6–2639**; none reaches 75 with **any** of 1–5 2D steps.
Five 2D steps at index 31 lose all 25 selected members; at index 36 all
25 survive but five steps still end at misfit **418.5**, showing that solver
survival alone does not cure the mismatch.

Threshold 25 additionally escalates indices 29–30, where the ten-step 1D
result already meets 75. Every tested threshold from **50 to 250** makes
the **same** decisions: it escalates 31–40, and none of those handoffs gives
an acceptable 2D result. Therefore these thresholds are observationally tied
and no unique optimal gate threshold can be identified. The original saved
post-1D benchmark reference RMSE was **0.66354**; ten-step 1D-only is
**0.64538**, but it still misses the data target at ten points. The only
fully scored 2D candidate with two steps is worse in RMSE and misfit and has
only two surviving members at its weakest point.

**Scope:** This is an independent, matched-prior gating experiment, not a
validated sequential 25–40 posterior, nor proof that a different 2D prior,
solver, or more than five 2D updates would fail. Truth is used only for
evaluation, never assimilation or member selection. Source inversion/HPC
files already contained uncommitted user edits at launch; provenance hashes
at study completion are `run_DA_sequential.py` d51064c46ec27f0251f2f2be399f53872ab3a233576b8e90a328c8e1f77d4cac,
`esmda_core.py` 1749f835137356001dff5ee2b0613e4a5c0683d60b59ba72e2aa01d3893a3def,
and `esmda_remote2d.py` 4638352591bbe8bb51af8036169030490a25e98d11256293cd6d9f9bf9c366c9.

Per-point survivor identities, failure counts and original-covariance
misfits: `runs/transition-mda-261016-2112/point-25.json` through
`point-40.json`; full threshold grid and pointwise table:
`runs/transition-mda-261016-2112/summary.md` and `summary.json`.
Relevant guard suite: **69 passed**. The metric loop rejected the
threshold-75/two-step candidate; threshold-75/five-step has no score because
of the index-31 failure. No discarded parameter candidate was a separate
source-code commit, so the study infrastructure and measured evidence remain.
The requested default of 25 source-code optimization iterations was not
performed: the six-threshold/five-step parameter grid was completed instead,
and no tested 2D configuration met the data-misfit requirement. The TSV logs
the two actual metric checks rather than inventing extra code iterations.
