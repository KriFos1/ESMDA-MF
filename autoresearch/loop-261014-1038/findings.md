# Transition-zone 0D/1D/2D ES-MDA + LM/Laplace study

## Goal and metric

Find a convergent configuration for Benchmark-3 transition indices 30–35 that
reduces both data misfit and Rh/Rv field RMSE against benchmark truth.

The numeric objective is lower-is-better:

```text
score = sqrt((candidate mean original-covariance reduced chi-square / 315.0210)
             * (candidate pooled Rh/Rv log10 RMSE / 0.6529911))
```

The fixed denominators are the six-index baseline. The RMSE is computed over
the 2D Rh and Rv field, cell-averaged from Benchmark-3 truth into each saved
window. A candidate with any nonconverged or missing MAP receives a `1e6 + n`
success penalty. The evaluator is `score_transition.py`; it recomputes
misfit using the original `data.pkl` and `var.pkl`, even for candidates trained
with altered variances.

## Best tested configuration

The best observed score is **0.48876** (51.1% below the baseline score of
1.0). This is a warm-resumed segment at index 31 from the preserved index-30 checkpoint and
processes indices 31–35 with seed 2026 (effective resume seed 2057), a 33×5×80
window, 400 members, Gaussian prior, Rh correlations 10/75 ft, latent
correlations 80/35 ft, ratio bounds 1–3, Rv cap 80 Ω·m, threshold 75, and TV
off, with `--tv-max-step .5`. The best-tested settings are
`--one-d-budget .25` and `--lm-laplace-seconds 1800`, with 15 LM iterations
and five ranked 2D starts.

**Comparison caveat:** the original baseline run used `tv_max_step=.25`, but
the autoresearch candidate commands omitted `--tv-max-step` and therefore used
the current CLI default `.5`. Consequently, the score difference from the
original baseline is an observed result for the candidate configuration, not
an isolated causal estimate of the 1D-budget/time changes. The 900→1800-second
comparison is controlled against iteration 1 because both candidates used
`tv_max_step=.5`.

The 0D schedule spends 0.271 cumulative MDA weight before crossing the nominal
0.25 cap, so a 0.25 1D budget admits no 1D ES-MDA update. The controller still
computes the 1D response and uses it for the fidelity gate. Under the best
configuration, all five new MAPs converged and selected 2D; index 30 remains
the common fixed baseline checkpoint.

| Index | Baseline fidelity | Baseline χ² | Baseline Rh/Rv RMSE | Best fidelity | Best χ² | Best Rh/Rv RMSE |
|---:|---|---:|---:|---|---:|---:|
| 30 | 1D | 24.01 | .431 / .403 | 1D (fixed) | 24.01 | .431 / .403 |
| 31 | 2D | 945.34 | .525 / .480 | 2D | 196.56 | .454 / .408 |
| 32 | 1D | 295.87 | .637 / .595 | 2D | 150.58 | .562 / .508 |
| 33 | 1D | 195.54 | .712 / .656 | 2D | 130.14 | .595 / .540 |
| 34 | 1D | 117.66 | .802 / .738 | 2D | 7.64 | .629 / .567 |
| 35 | 1D | 311.71 | .877 / .780 | 2D | 3.01 | .870 / .752 |

Across the six points, mean reduced χ² falls **315.02→85.32** and pooled
Rh/Rv RMSE falls **0.6530→0.5759**. The index-35 2D solution converged after
extending the shared optimization budget: 3 of 5 2D starts converged. The
candidate results and complete remote receipts are retained in
`runs/autoresearch-261014-1038-i03-seconds1800/`.

## Controlled autoresearch iterations

| Iteration | Change | Score | Outcome |
|---:|---|---:|---|
| 1 | 1D budget .75→.25 | .60404 | Keep. All five new MAPs converged; indices 31–34 selected 2D; index 35 initially fell back to a converged 1D MAP. |
| 2 | 1D budget .25→.32 (one 1D MDA increment) | 1,000,001 | Discard. Index 34 ended `no_acceptable_trial`; the unconstrained score was .66281. |
| 3 | LM/Laplace time 900→1,800 s | .48876 | Keep. Index 35 obtained a converged 2D MAP. |
| 4 | Multiply UADP/UADA variances by 4 | .89551 | Discard. Field RMSE improved to .5082 but original-covariance mean χ² rose to 324.58. |
| 5 | Multiply UADP/UADA variances by 2 | 1.40229 | Discard. Field RMSE improved to .5589 but mean χ² rose to 723.70. |

The 1D-stage update at budget .32 is not a safe improvement: it gave a slightly
worse data misfit at index 32 and failed to converge at index 34. UADP/UADA
downweighting shows a Pareto tradeoff but is not supported by the joint
misfit/RMSE objective. Keep the original observation covariance for the
selected configuration.

The baseline component-wise reduced residual means (indices 30–35) were
UADP 536, UADA 518, UHRP 159, UHRA 390, UHAP 180, and UHAA 107. These are
diagnostics of mismatch, not proof that the higher-residual components are less
informative. Downweighting UADP/UADA substantially worsened original-covariance
fit, reinforcing that distinction.

## Prior, 2D-start, and TV evidence

- Existing transition-region prior screens found no monotone or globally
  dominant lateral-correlation setting. In the saved point-31 screen,
  Lx=10 ft with latent scale 1 achieved a 15-step 2D misfit of 194.4, while
  Lx=20 ft / latent scale 1 reached 691.7 after six accepted steps. See
  `Investigation.md` and `runs/benchmark3-point31-lateral-ablation-r3-cap80-20261013*`.
- Existing prior-family tests compared blocky, Gaussian, and long-block mixture
  priors across conductive, reservoir, and fault/transition zones; each won on
  different point metrics, so none was promoted globally. See
  `PRIOR_FAMILY_ASSESSMENT.md`.
- An x-invariant 2D extension at index 32 gave a solver-compatible starting
  response and an interrupted LM path reached χ² 21.70, but the run stopped
  before a saved converged posterior. At index 31, invariant-start fields with
  deep high-Rv sheets still triggered the solver failure. See
  `INDEX32_2D_FAILURE_ASSESSMENT.md` and `SOLVER_BLOCK_ROM_DEBUG.md`.
- Prior TV pilots found improved 1D transition recovery at TV=.2, but forced
  single-update 2D results had a nonlinear linearization gap and the current
  LM/Laplace path requires TV off. The best transition result here was obtained
  without TV; no new TV penalty was needed. See `TV_PILOT_ASSESSMENT.md`.

## MDA-family survey

`mda-variants.md` records the relevant families and references: stochastic
ES-MDA, adaptive inflation schedules, IES/EnRML and LM-IES, deterministic
IEnKS/square-root methods, and annealed-importance/tempered smoother hybrids.
The current system is a hybrid workflow: stochastic ES-MDA at lower fidelities,
a pre-final best-member 1D/2D gate, then adjoint 2D MAP optimization and
Laplace sampling. It is not a full 2D IES/EnRML method. The gate and stage
budget are more consequential here than swapping in a different ensemble
update formula before the 2D start/convergence path is stable.

## Decision

Use the **0.25 1D budget + 1,800-second LM/Laplace budget + original
measurement covariance** for the tested transition segment. This achieved a
complete, converged six-index result under the stated joint metric. The method
uses 1D simulation for gating but intentionally performs no 1D ES-MDA increment
at these settings; the attempted single 1D increment arm did not converge at
index 34. The source tree was already heavily modified before this work; only
the isolated autoresearch configuration commits were staged/committed, and no
production source files were altered by this study.
