# Unified-domain pilot assessment — 2026-10-06

Pilots on branch `2D-curtain` (artifacts `runs/unified-pilot-20261006T093911Z`,
`runs/unified-pilot2-20261006T103314Z`). Controls identical to the stage-TV
pilots: seed 2026, threshold 100, blocky prior, projection off, TV
`1D 0.1 / 2D 0.05 / step 0.25`, cap 8. Baseline: the staged r2 partial run
(130 checkpoints, nx=15). Zone metrics: bit-column ensemble mean vs
cell-averaged truth, log10 RMSE.

## 1D regime (gate not tripped)

| Zone | r2 (staged, nx15) | u15 | u33 (250) | u33 (400) |
|---|---|---|---|---|
| 27–29 | 0.107 | 0.11–0.28 | 0.11–0.36 | 0.21–0.31 |
| 80–82 | 0.546 | **0.38–0.44** | 0.37–0.46 | 0.36–0.44 |
| 160–162 | (no r2) | 0.60–0.79 | 0.81–0.95 | **0.45–0.56** |

- The unified 1D regime matches or beats the staged baseline at the reservoir
  zone and degrades gracefully at the transition zone.
- **Rank is the binding constraint at nx=33**: with 250 members the larger
  window is *worse* than nx=15 at hard zones; with 400 members it is the best
  configuration overall. Ensemble size must scale with window size.
- No-update drift control (`drift_control.py`): spread stationary
  (1.52→1.55 ln), reservoir fraction 0.39→0.35, lateral correlations stable
  across all 281 shifts — the conditional edge machinery alone does not
  degenerate the ensemble.
- Operational: parallel controller waves must pin BLAS threads to 1
  (`OMP_NUM_THREADS=1`); unpinned 9-way waves thrash and miss 900 s budgets.

## 2D engagement at index 80 (forced)

| Engagement prior | Substitutions | Stored misfit | **Forward-verified** | RMSE | Time |
|---|---:|---:|---:|---:|---:|
| staged (r2 best, reference) | 0–8 | ~9–4 | **37.4** | 0.534 | ~95 s |
| unified ensemble, single-update | 8 | 49905 | 51966 | 0.469 | 987 s |
| unified ensemble, iterated LM (3) | 5 | 63239 | 63239 | 0.445 | 1297 s |
| unified ensemble, nx33/ne400 | 1 | 100479 | 114958* | 0.495 | 470 s |
| **unified x-invariant** | **0** | **15.5** | **27.9** | **0.432** | **88 s** |

*only 2 of 5 member solves completed (timeouts); magnitude representative.

Findings:

1. **Engaging 2D directly from the carried ensemble is structurally broken.**
   The 1D evidence cannot constrain flanks, so member flank realizations are
   individually orders of magnitude from 2D-data consistency (prior forward
   misfits 1e4–2e5). Neither one trust-bounded linearized step nor three
   iterated LM iterations bridges that (verified ~5e4–1.2e5), and the sharp
   blocky flanks also crash the solver (substitution carousels, 8-cap).
   The staged design never exposed this because its kriging flank priors
   pre-smoothed the lateral structure.
2. **The x-invariant engagement prior fixes it.** Extending each member's bit
   column laterally (`--engagement-prior invariant`,
   `engagement_invariant_windows`) claims nothing beyond the 1D evidence:
   prior forward misfits 77–101, solver-friendly (zero substitutions), one
   batch, 88 s point time, and the **best verified fit measured at this
   point (27.9 vs staged 37.4)** with the best RMSE (0.432).
3. Cap + 1D fallback remained correct throughout (u33-80 abandoned to a
   healthy 1D posterior; no stalls, no hangs).

## Recommended configuration

Unified domain, `--nx 33 --ensemble-size 400`, TV `1D 0.1 / 2D 0.05 / step
0.25`, `--engagement-prior invariant`, threshold 100, cap 8. Open before a
full run: none blocking; optionally confirm u33+400+invariant at index 80 in
one forced diagnostic.

## Confirmation at the recommended configuration

`runs/unified-confirm-20261006T131623Z/` — forced index 80 with
`--nx 33 --ensemble-size 400 --engagement-prior invariant`:

- Prior forward misfits 88–133 (invariant prior consistent, zero substitutions).
- Linearized posterior misfits 6.2–13.0; **forward-verified mean 35.7**
  (all 5 members, 34 s check) vs staged 37.4 and u15+invariant 27.9.
  Note the printed `gaussian_diagnostic_loss` (~3e7 here) is the
  inflated-covariance plus prior-penalty diagnostic, not the data misfit.
- The u15 variant fits this point slightly better but carries the
  ±35 ft window-truncation error (up to 98σ, window-size study); the
  recommended configuration trades that for boundary correctness at
  near-equal per-point fit.

No blockers remain for a full unified run with: `--nx 33
--ensemble-size 400 --engagement-prior invariant`, TV `1D 0.1 / 2D 0.05 /
step 0.25`, threshold 100, cap 8.

## Caveats

One seed; three-point zones; one forced geometry; the nx=33 choice rests on
the window-size study at two indices (both fully covered at ±80 ft). Zone-160
has no staged baseline (r2 stopped at 130).
