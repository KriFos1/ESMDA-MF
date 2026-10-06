# Projected-prior pilot assessment

## Corrected, matched follow-up (2026-10-05)

Artifacts: `runs/projection-corrected-20261005T103027Z/assessment.json`.
All four matched pilots completed. Each covers three points: 27–29 or 80–82.
Both arms use natural-log coordinates converted from documented log10 CLI
levels, exactly 1.5/30 Ohm m modes, 0.10 log10 within-mode width, identical
blocky samplers/chaining/seed/member count, threshold 100 and single-update
2D. Only `--prior-projection on/off` differs (besides artifact paths).

| Zone | Projection | Mean RMSE (log10 units) | Near-truth fraction | Verified mean posterior misfit | Mean point time | 2D points |
|---|---|---:|---:|---:|---:|---:|
| 27–29 | off | 0.209 | 79.0% | 45.84 | 1.33 s | 0 |
| 27–29 | on | 0.167 | 94.7% | 31.40 | 1.30 s | 0 |
| 80–82 | off | 0.382 | 70.0% | 62.10 | 4.60 s | 0 |
| 80–82 | on | 0.776 | 63.7% | 93.53 | 33.00 s | 1 |

Projection improved transition-zone recovery but worsened reservoir-zone
recovery. At index 80 it triggered 2D (1D final misfit 142.2 > 100), whose
92.9-second point duration missed the one-minute target; the subsequent 1D
columns retained the poorer recovery. This coupled pipeline comparison does
not isolate the effect of projection from its effect on fidelity selection.
It is one seed and a short trajectory segment, not a production acceptance
test. No full run was launched.

An independent forward-only batch evaluated all five saved posterior members
at on-80/index 80: mean true forward misfit 95.31, versus the explicitly
linearized stored estimate 91.40 (per-member forward range 69.56–153.97).
All solves succeeded; the validation took 38.95 s. Evidence is recorded in
`check-on-80/report.json`, including the checkpoint SHA-256. Off-arm points
were all 1D and already had forward-evaluated posterior predictions.
The verified table substitutes that forward check for the index-80 linearized
estimate, then averages the per-member reduced misfits across the three points.

The follow-up used an eight-minute cap on the concurrent pilot wave, a
two-minute cap on validation, one-minute Slurm jobs and a 45-second validation
solver timeout. No cap was reached. The `pilot_projection.py` harness stops
its own process groups and cancels only its own batch IDs on timeout.

**Decision:** do not adopt mandatory hard projection or relaunch the full
inversion from these results. Projection off is preferable in the tested
reservoir segment, but selecting a spatially varying rule from these short
tests would be premature. Prior sampling still does not impose a posterior
TV penalty; a hard modal projection is not equivalent to TV regularization.

## Earlier, incorrectly parameterized pilots (retained for provenance)

## Verdict

Do not relaunch the full inversion. Both projected-prior pilots completed, but
the reservoir-zone pilot fails the data-fit assessment. A log-base error in the
prior configuration also means these runs did not test the intended physical
resistivity modes.

## Parameterization correction

The forward paths use natural-log resistivity: EMsim exponentiates its `rh`
and `rv` inputs, and `pxwf.execution.worker.write_inputs` uses `np.exp(field)`.
The prior arguments were described as log10 and passed directly to these paths.
Thus the values 0.18 and 1.46 represented 1.197 and 4.306 Ohm m, rather than the
intended background near 1.5 Ohm m and reservoir near 30 Ohm m. Exact natural-log
levels for those two physical values are 0.405465 and 3.401197. A width of 0.10
log10 units corresponds to 0.230259 natural-log units.

Earlier assessment scripts also compared natural-log posterior parameters with
log10 truth. Their RMSE and edge-amplitude claims are invalid and must not be
used to select a prior. The table below uses natural logs for both arrays, then
divides errors by ln(10) to report RMSE in log10 units.

## Corrected pilot comparison

Truth is sampled at the nearest benchmark x-cell to the logging point, and
averaged in natural-log space over each local 5-foot z-cell. For a 2D checkpoint
the comparison uses the bit column. Near-truth fraction counts member cells
within 0.30 log10 units of the corresponding averaged truth cell. RMSE is for
the ensemble-mean log-resistivity column.

| Zone | Configuration | Mean RMSE (log10 units) | Near-truth fraction | Mean 1D final misfit | Mean reported posterior misfit | Elapsed |
|---|---|---:|---:|---:|---:|---:|
| 27–34 | Gaussian baseline A | 0.315 | 55.0% | 171.7 | 155.7 | 291.7 s |
| 27–34 | Projected blocky B3 | 0.250 | 64.0% | 180.3 | 487.6 | 413.1 s |
| 80–89 | Gaussian baseline A | 0.606 | 53.7% | 148.8 | 567.4 | 1151.1 s |
| 80–89 | Projected blocky B3 | 0.653 | 29.5% | 2153.7 | 88486666.3 | 1125.2 s |

Artifacts: `runs/pilotAB-A-27-20261005`, `runs/pilotAB-A-80-20261005`,
`runs/pilotAB-B3-27-20261005`, and `runs/pilotAB-B3-80-20261005`.
Both B3 runs exited with code 0 and recorded zero substitutions.
Their average elapsed times were 51.6 and 112.5 seconds per point, respectively;
the second zone does not meet the one-minute target.

These are exploratory comparisons: A and B differ in prior family, chaining,
projection and 2D prior preparation. They do not isolate projection alone.

## Independent local forward control

The local UTA1D control uses both true Rh and true Rv from the same benchmark
column, geometric-cell averages, each checkpoint's window shift, and the
checkpoint's retained observation channels with the original measurement
covariance. The true anisotropy ratio is converted to the latent coordinate
using the inverse normal CDF of ln(Rv/Rh)/ln(5).

| Index | Truth-derived 5-foot-column 1D misfit | B3 final 1D misfit |
|---|---:|---:|
| 31 | 622.39 | 119.44 |
| 84 | 7.44 | 2348.87 |

At index 84 the truth-derived column fits substantially better. The claim that
the high B3 misfit is necessarily an unavoidable lateral-fidelity error is not
supported. Index 31 shows why low data misfit alone is insufficient to establish
correct model recovery. This averaging control is not an exact coarse-grid
equivalent of the fine model, but provides a feasible forward comparison.

## Reporting limitation

`update_members_once` stores `f(initial) + J @ (posterior - initial)` as the
posterior prediction. It does not execute the forward simulator at the saved
posterior. All 2D posterior misfit entries in these single-update pilots are
linearized estimates, not measured posterior forward misfits. In particular,
the very large B3 values cannot establish how the saved posterior actually
predicts the observations. They indicate that the single-step linearized result
is unsuitable as a convergence certificate.

## Before another pilot

1. Correct the prior-unit interface, including widths and projection radius;
   explicitly record physical mode values and the internal natural-log units.
2. Distinguish forward-evaluated and linearized predictions in checkpoint and
   plotting metadata. Do not call the latter verified posterior data fits.
3. Compare corrected blocky runs with projection on/off under otherwise
   identical configuration, without deleting the pilot artifacts.
4. Perform a small, explicitly budgeted posterior forward validation and retain
   the no-full-relaunch gate until fit, recovery and elapsed-time criteria pass.
