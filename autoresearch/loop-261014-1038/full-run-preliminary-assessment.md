# Preliminary full-run assessment — 2026-10-15

Run: `runs/benchmark3-full-mda-laplace-b025-t75-lm1800-step050-20261014/`

The user stopped the full 0–280 run at index 43. It saved checkpoints 0–42;
index 43 had completed its 0D updates but had no final-stage result or
checkpoint. Supervisor return code -15 and status `FAILED` record that
intentional stop; the Slurm queue is empty. The confidence curtain had rendered
43 points at the stop. This remains a preliminary assessment, not a completed
full-run result.

Across completed indices 0–42, 39 MAPs selected 1D and four selected 2D; index
25 is the sole nonconverged MAP (`no_acceptable_trial`). Mean reduced χ² is
67.66 and pooled Rh/Rv log10 RMSE is 0.449 across these varying geological
regions; these broad averages are descriptive, not a like-for-like score.

| Index | Selected result | χ² | Rh/Rv RMSE log10 | Note |
|---:|---|---:|---:|---|
| 36 | 1D fallback | 706.64 | 1.008 / .812 | No converged 2D start |
| 37 | 1D fallback | 535.78 | 1.001 / .788 | No converged 2D start |
| 38 | 1D fallback | 392.72 | .968 / .744 | No converged 2D start |
| 39 | 1D fallback | 199.82 | .948 / .723 | No converged 2D start |
| 40 | 1D fallback | 205.54 | .893 / .696 | No converged 2D start |
| 41 | 1D fallback | 110.19 | .855 / .679 | No converged 2D start |
| 42 | 1D fallback | 71.76 | .826 / .694 | No converged 2D start |

## Transition-window comparison

Both comparisons use the original measurement covariance and the same
truth-aligned Rh/Rv log10 RMSE evaluator. The autoresearch winner is the warm
resumed segment candidate; the full run starts at index 0 with seed 2026.

| Index | Full-run fidelity | Full-run χ² | Full-run Rh/Rv RMSE | Autoresearch-best χ² | Autoresearch-best Rh/Rv RMSE |
|---:|---|---:|---:|---:|---:|
| 30 | 1D | 13.64 | .412 / .371 | 24.01 | .431 / .403 (fixed reference) |
| 31 | 2D | 67.66 | .475 / .429 | 196.56 | .454 / .408 |
| 32 | 2D | 43.13 | .504 / .453 | 150.58 | .562 / .508 |
| 33 | 2D | 140.19 | .588 / .525 | 130.14 | .595 / .540 |
| 34 | 1D fallback | 83.82 | .656 / .599 | 7.64 | .629 / .567 |
| 35 | 2D | 311.55 | .774 / .677 | 3.01 | .870 / .752 |

Over indices 30–35, the full-run composite is **0.54281**, compared with
**0.48876** for the autoresearch winner. Its mean reduced χ² is **110.00**
(autoresearch: **85.32**), while its pooled field RMSE is **0.5510**
(autoresearch: **0.5759**). Thus, the live full-run segment has somewhat better
field RMSE but worse data fit than the best warm-resumed candidate; it still
improves both aggregate metrics over the initial baseline (χ² 315.02, RMSE
0.6530).

The main differences are at indices 34–35: index 34 had no converged 2D start
and fell back to 1D (χ² 83.82), while index 35 converged at 2D but retained
χ² 311.55. At index 36, the 2D gate also fell back to a converged 1D MAP:
χ² 706.64, Rh/Rv RMSE 1.008/0.812. The previous threshold-75 run at index 36
had χ² 622.02 and RMSE 1.025/0.852. Index 25 is the one non-converged MAP so
far (`no_acceptable_trial`), despite its low χ² of 0.785.

Across indices 0–36, 33 MAPs selected 1D and four selected 2D; all but index 25
converged. The aggregate across this wider span is descriptive only because
data difficulty and geology vary along the trajectory.

## Interpretation and caveats

- The full run starts from index 0 and uses the same random seed for the full
  trajectory; the successful autoresearch segment warm-resumed from index 30
  and used a fresh resume seed. These are not identical stochastic paths.
- The earlier baseline used `tv_max_step=0.25`; the tested winner and this full
  run use `0.5`. The earlier transition comparison therefore changed more than
  the 1D budget and LM time. The full run reproduces the actual winning run
  manifest, but it does not isolate the effect of each parameter.
- This is an early trajectory check, not a full-run result. Continue monitoring
  the saved checkpoints and live curtain before drawing a final conclusion.

## Snapshot evidence

- Run manifest/status: `runs/benchmark3-full-mda-laplace-b025-t75-lm1800-step050-20261014/run.json` and `.status.json`
- Controller log: `runs/benchmark3-full-mda-laplace-b025-t75-lm1800-step050-20261014.log`
- Curtain: `runs/benchmark3-full-mda-laplace-b025-t75-lm1800-step050-20261014/curtain.png`
- Autoresearch transition summary: `autoresearch/loop-261014-1038/findings.md`
