# TV-regularized multi-fidelity pilot

## Implementation

`tv_regularization.py` implements smoothed isotropic TV of **ln Rh**, not the
anisotropy latent. Natural-log units are used internally. Forward differences
use physical spacing in feet. Outer boundaries have zero forward difference;
internal fixed/free interfaces are included. The 1D penalty is integrated along
depth; 2D integrated TV is divided by window width, so an x-invariant 2D field
has the same penalty as its 1D column. Epsilon defaults to 0.001 ln units/ft.

After each 1D ES-MDA update, each member receives a TV proximal step computed
by bounded-count IRLS linear solves. The final 1D update receives the remaining
dose. At the 2D stage, inner IRLS uses the existing Jacobian, a weak independent
amplitude penalty instead of the correlated Gaussian regularization, and
box/trust bounds on active parameters. Default maximum increment is 0.5
optimizer-coordinate units. No inner IRLS sweep calls the forward simulator.

For each point, the 1D/2D TV fractions sum to one after excluding 0D assimilation
weight. Their strengths sum to `--tv-weight`, regardless of the number of MDA
iterations or the selected final fidelity. This is an approximate regularized
ensemble algorithm, not exact sampling of a Bayesian TV posterior. Hard mode
projection is disabled in TV mode. The initial blocky prior, chaining and
existing 2D column preparation are retained, so TV is not the sole source of
structure. TV currently requires the fixed-Jacobian `--single-update` mode.

Historical `posterior_losses` remain Gaussian diagnostic values for checkpoint
compatibility; TV checkpoints explicitly label them `gaussian_diagnostic` and
store regularization history. Data misfits, computed with original measurement
covariance, are the assessment quantities. 2D single-update posterior predictions
remain explicitly **linearized**, not forward verified.

## Numerical checks

47 tests passed before the pilot, including:

- TV derivative agreement with central differences;
- zero penalty for a constant field and matching 1D/x-invariant-2D penalties;
- spacing/refinement consistency;
- proximal objective decrease, preservation of a large jump and removal of
  small oscillations without mode snapping;
- unchanged latent coordinates and retained member differences;
- fixed-cell/trust-bound behavior in the 2D local solve;
- identical total TV dose with different MDA iteration counts;
- one batched forward/Jacobian evaluation for the single-update inner solver.

## Pilot design and artifacts

Root: `runs/tv-pilot-20261005T111823Z/`.
`assessment.json` contains the original eight unforced pilots, two forced
diagnostics and their forward checks. An additional forced strength-0.2
diagnostic is preserved separately in `forced-tv020-80/` and
`check-forced-tv020-80/report.json` under the same root.

Unforced comparisons cover points 27–29 and 80–82 with seed 2026, 250 ensemble
members, five selected members, threshold 100, strict chaining, corrected
1.5/30 Ohm m prior levels, projection off and single-update 2D. Cases are the
existing baseline, zero TV, TV=0.05 and TV=0.2. The zero-TV case reproduces
baseline 1D parameters and numerical results exactly; in 2D it controls for the
new weak independent prior/trust solve.

All unforced points selected 1D. Three additional **forced**, single-point
diagnostics at index 80 exercise 2D with zero TV, 0.05 and 0.2. These are
diagnostics, not evidence that the unforced gate should have selected 2D.
The nonzero-TV forced cases also have TV in their preceding 1D updates, so their
prior fields are different; their outcomes do not isolate a 2D-only TV weight.

Pilot wave cap: 480 s. Forced diagnostic and check wave caps: 120 s. Slurm jobs
have one-minute limits; independent checks have a 45-second solve timeout. All
pilots and all three five-member posterior checks completed without timeouts,
solver substitutions or full rerun. Baseline/TV timings are concurrent-wave
measurements, not isolated CPU benchmarks.

## Results: unforced, forward-evaluated posteriors

RMSE is for the ensemble-mean bit column against geometric-cell-averaged truth,
expressed in log10 units. Data misfit is the mean of member reduced chi-square
values using original covariance and the existing retained-channel mask.

| Zone | Setting | Mean RMSE (log10) | Forward mean misfit | Mean column TV | Mean point time |
|---|---|---:|---:|---:|---:|
| 27–29 | baseline / zero TV | 0.209 | 45.84 | 11.44 | 1.69 s / 1.86 s |
| 27–29 | TV 0.05 | 0.207 | 33.52 | 7.94 | 2.56 s |
| 27–29 | TV 0.2 | **0.147** | **18.41** | 4.57 | 2.47 s |
| 80–82 | baseline / zero TV | 0.382 | 62.10 | 13.46 | 6.23 s / 6.13 s |
| 80–82 | TV 0.05 | 0.383 | **49.48** | 13.00 | 8.11 s |
| 80–82 | TV 0.2 | **0.354** | 49.98 | 10.79 | 8.17 s |

TV 0.2 reduces transition-zone RMSE by about 30% and forward misfit by 60%.
In the reservoir zone, its RMSE improves about 7% and forward misfit about 20%.
Member spread is reduced but remains nonzero (transition 0.385→0.312 ln units;
reservoir 0.158→0.143).

### Edge structure

For jumps above 0.30 log10 per cell, transition-zone truth has zero edges; the
baseline averages 5.53 such edges/member and TV 0.2 averages 1.80. Reservoir
truth has two; baseline averages 6.93 and TV 0.2 averages 5.73. The reservoir
maximum jumps remain about 0.99–1.29 log10 under TV 0.2, rather than collapsing
to a uniform field. TV reduces false variation, but interface recovery is not
yet complete. These counts do not establish accurate interface locations.

## Results: forced 2D with independent posterior forward checks

| TV weight | Mean linearized misfit | Mean actual forward misfit | Column RMSE (log10) | Point time |
|---|---:|---:|---:|---:|
| 0 | 4.90 | 60.52 | 0.546 | 92.04 s |
| 0.05 | 4.26 | **45.50** | 0.534 | 95.58 s |
| 0.2 | 2.94 | 90.80 | **0.517** | 94.53 s |

Each inversion diagnostic uses exactly five field evaluations (one Jacobian
batch), with no substitutions. The independent forward checks use a separate
five-member forward-only batch and take about 39 s each. Checks are tied to
checkpoint SHA-256 hashes and preserve raw predictions in `forward.npz`.

The nonlinear forward fit is much worse than the linearized prediction, even
with the bounded step. Stronger TV improves column recovery here but does not
monotonically improve nonlinear data fit. The 0.05 case has the best verified
2D fit among these three diagnostics. All forced points still exceed the
one-minute goal. Extra inner TV sweeps themselves add no simulator batches;
the one-batch remote execution/queue overhead remains the runtime limitation.

## Stage-specific weights and trust bound — 2026-10-05 (afternoon)

Follow-up pilots (`pilot_tv2.py`; artifacts `runs/stage-tv2-…Z`, `stage-tv3-…Z`,
`stage-tv4-…Z`, `stage-tv5-…Z`) answered the two open questions: stage-specific
TV weights and the linearization gap. New controls: `--tv-weight-1d/--tv-weight-2d`
(default to `--tv-weight`; the per-point dose sums to the stage weight actually
used, recorded as `tv_weights_effective`), and pilot polling cap / sequential
scheduling / 2-minute walltime (see HPC_TESTING.md).

Matched forced-2D diagnostics at index 80, seed 2026. Runs sharing one 1D weight
have byte-identical 1D prefixes (same seed), so only the 2D controls differ;
runs with different 1D weights also differ in their 2D starting ensemble — that
is the quantity being compared. "Tight" = `--tv-max-step .25` vs the default .5.

| 1D→2D weight, step | Linearized misfit | **Forward-verified misfit** | RMSE (log10) | Max jump |
|---|---:|---:|---:|---:|
| .05→.05, .5 (prior pilot) | 4.26 | 45.50 | 0.534 | — |
| .05→.05, .25 | 11.63 | 38.03 | 0.484 | 0.84 |
| **.1→.05, .25** | 10.38 | **37.42** | **0.464** | 0.85 |
| .2→0, .5 | 2.94 | 95.94 | 0.517 | 1.08 |
| .2→.05, .5 | 2.93 | 94.52 | 0.517 | 1.08 |
| .2→.05, .25 | 9.29 | 75.79 | 0.461 | 0.86 |
| .2→.1, .25 | 9.29 | 72.36 | 0.460 | 0.86 |

Findings:

1. **The loose 0.5 trust bound was over-trusting.** It reports linearized
   misfit ≈ 2.9 while the true misfit is ≈ 95. Tightening to 0.25 raises the
   stored (honest) linearized estimate to ≈ 9 and *improves* the verified
   forward misfit in every matched pair (94.5→75.8; 45.5→38.0), with lower
   RMSE and fewer false jumps.
2. **The 1D-stage dose controls the 2D starting ensemble quality.** With 1D
   TV 0.2 the engaged-2D verified misfit is 72–96 regardless of 2D weight;
   dropping the 1D weight to 0.05–0.1 reaches 37–38. Heavy 1D smoothing helps
   1D recovery but leaves single-update 2D points a poor starting ensemble.
3. Unforced 1D behavior of weight 0.1 (zones 27/80/160, three points each,
   `stage-tv5`): better RMSE than no-TV on all nine indices, intermediate
   between no-TV and 0.2 as expected; false jumps reduced everywhere.

**Best tested configuration: `--regularization tv --tv-weight-1d 0.1
--tv-weight-2d 0.05 --tv-max-step 0.25`** — best verified engaged-2D misfit
(37.4) and RMSE (0.464) among all tested cells, and clearly better than
no-TV on 1D recovery. Open before a full rerun: single seed, one forced
geometry, short 1D segments; 2D engaged points cost 1.5–2 min under load
(batch solve time dominates; the polling cap removed only detection
latency). The full-run decision remains with the user.


