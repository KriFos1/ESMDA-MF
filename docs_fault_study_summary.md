# Fault / 600 ft transition memory study

Script: `run_fault_study.sh` (variants `base | ahead | mean | latent | all`).
Runs: `runs/fault-<variant>-g25-s100`, logging points 100-140, seed 2031.

## Setup (identical in all runs)
- Sequential 1D ES-MDA (`sequential_matern_1d.py`), 250 members, 10 MDA steps, linear-inflation tempering, EnIF analysis.
- Matern prior, `log_clipped` anisotropy (mean 0.5, std 0.5), `--memory-ft 50`.
- Probe-and-revert gate at post-1D member-median chi2/n > 25 (`--on-gate stop`); gated points go to the 2D Laplace/MAP stage on the HPC.
- Reference: `runs/benchmark3-reference-bedding`.

## How the memory works
Between points the carried ensemble is relaxed toward the prior:

    new = prior_mean + rho_mean*(mean - prior_mean) + sqrt(1 - rho^2)*noise

`rho = exp(-step/memory_length)`; `rho_mean` uses the same formula with `memory_ft * mean_factor`.

| Variant | Options | Meaning |
|---|---|---|
| base | none | uniform 50 ft memory (control) |
| ahead | `--memory-ahead-factor 0.5 --memory-surprise-factor 0.3` | shorter memory for cells ahead of the bit and, scaled by the previous point's misfit (chi2/n 3 -> 25, log scale), for the bit neighbourhood |
| mean | `--memory-mean-factor 4` | ensemble mean keeps memory of 200 ft; spread still 50 ft |
| latent | `--memory-latent-factor 3` | anisotropy latent block has 150 ft memory |
| all | all of the above | |

## Results
- **Gate behaviour identical:** every variant opens the gate at 103-133 and 140 (32 points -> 2D). Post-1D chi2/n peaks at ~300-700 (up to ~1700 for mean/all at 115-117), far above 25, so no memory option avoids the 2D stage.
- **Non-gated points** have post-1D chi2/n of about 9-21; gated points start at 25-34 (103, 140).
- **Rh RMSE (log10), mean over the 32 2D points:**

| Variant | RMSE | vs base |
|---|---|---|
| base | 0.327 | - |
| ahead | 0.321 | -2% |
| latent | 0.326 | ~0% |
| mean | 0.312 | -5% |
| all | 0.302 | -8% |

- `mean` gives better priors at nearly every point from 111 to 133. `ahead` helps slightly (mostly 103-110). `latent` has no measurable effect.
- 2D MAP misfit (0.11-0.13), MAP iterations (~10-12) and run time (4.5-5.1 h) are similar.
- Cost of long mean memory: for `mean`/`all` the 1D fit just after the transition (115-117) is much worse than base (chi2/n ~1000-1700 vs ~138), because the retained mean is stale. The 2D stage recovers it.

## Caveats
- One seed, 250 members; differences are 0.01-0.03 in RMSE. Ranking of `ahead` vs `latent` is not established.
- Only Rh RMSE was scored; no uncertainty calibration or curtain comparison.

## Next experiments
### Mean factor sweep
Run `--memory-mean-factor` over **1 (= base), 2, 3, 4 (done), 8** (with `--memory-ft 50`). Logging points are 5 ft apart, so 50 ft is 10 points and 200 ft (factor 4) is 40 points, i.e. the whole 100-140 window; factor 16 (800 ft) would be effectively no relaxation and is not worth running. Scripted in `run_fault_study2.sh`, then, to disentangle the product, one run with `--memory-ft 100 --memory-mean-factor 2`. Suggested checks:
- Rh RMSE vs factor: expect a minimum; the 1D chi2/n at 115-119 will rise with the factor.
- Do 2D MAP iterations/time grow with the factor (stale prior)?
- Test with `--memory-ahead-factor 0.5 --memory-surprise-factor 0.3` on top, since `all` was the best.
- Repeat the best two factors and base with at least 2-3 seeds.
Note that `run_fault_study.sh` needs a new variant (or an `EXTRA` override) to take a factor, and `NAME` must include it to avoid overwriting.

### Gate threshold
25 is not shown to be optimal; it just sits in the gap between non-gated (<=21) and gated (>=25) points here, with little margin (index 102 at 21, index 140 at 25-34). Suggested sweep: `GATE=15, 25, 50, 100` plus `GATE=1.5` (all points to 2D) as the reference. Compare final RMSE, number of 2D points and cost (~8 min per 2D point on average, 4.5 h for 32 points). A higher gate saves 2D work but accepts 1D updates at poorly fitting points, which changes what is carried forward, so the result cannot be predicted from these runs.

### Ahead/surprise taper and strength
With 5 ft spacing the default `--memory-taper-ft 40` is only 8 points, so the ahead/surprise shortening acts over a short span, which may explain its small effect. Also, the surprise map (chi2/n 3 -> 25) saturates at 1 throughout the transition (chi2/n of 100s), so surprise mainly matters at its onset. Tested in `run_fault_study2.sh`: `best-taper80`, `best-taper150` (taper 80/150 ft with ahead 0.5 / surprise 0.3) and `strong-taper80`, `strong-taper150` (ahead 0.25 / surprise 0.1). Wider surprise scale (`--surprise-misfit 3 300`, so surprise keeps grading within the transition) is tested by `best-surp300`, `best-taper80-surp300` and `strong-taper80-surp300`.

 rank                                    run                   experiment  mean_factor  ahead_factor  surprise_factor  taper  gate  start  assim    metric  rel_to_best      metric_path                                                                   file    status  memory_ft  seed
    1           fault2-best-surp300-g25-s100                 best-surp300          NaN           NaN              NaN    NaN    25    100    110 0.0605966            1 posterior_misfit           fault2-best-surp300-g25-s100/inversion_results_assim_110.pkl        ok        NaN   NaN
    2 fault2-strong-taper80-surp300-g25-s100               strong-surp300          NaN           NaN              NaN     80    25    100    110 0.0635031      1.04797 posterior_misfit fault2-strong-taper80-surp300-g25-s100/inversion_results_assim_110.pkl        ok        NaN   NaN
    3   fault2-best-taper80-surp300-g25-s100                 best-surp300          NaN           NaN              NaN     80    25    100    110 0.0643917      1.06263 posterior_misfit   fault2-best-taper80-surp300-g25-s100/inversion_results_assim_110.pkl        ok        NaN   NaN
    4         fault2-strong-taper80-g25-s100                       strong          NaN           NaN              NaN     80    25    100    110  0.066782      1.10207 posterior_misfit         fault2-strong-taper80-g25-s100/inversion_results_assim_110.pkl        ok        NaN   NaN
    5                    fault-mean-g25-s100                   fault-mean          NaN           NaN              NaN    NaN    25    100    110 0.0668419      1.10306 posterior_misfit                    fault-mean-g25-s100/inversion_results_assim_110.pkl        ok        NaN   NaN
 
 rank                                    run                   experiment  mean_factor  ahead_factor  surprise_factor  taper  gate  start  assim    metric  rel_to_best      metric_path                                                                   file    status  memory_ft  seed
    1 fault2-strong-taper80-surp300-g25-s100               strong-surp300          NaN           NaN              NaN     80    25    100    115 0.0911616            1 posterior_misfit fault2-strong-taper80-surp300-g25-s100/inversion_results_assim_115.pkl        ok        NaN   NaN
    2   fault2-best-taper80-surp300-g25-s100                 best-surp300          NaN           NaN              NaN     80    25    100    115  0.093492      1.02556 posterior_misfit   fault2-best-taper80-surp300-g25-s100/inversion_results_assim_115.pkl        ok        NaN   NaN
    3           fault2-best-surp300-g25-s100                 best-surp300          NaN           NaN              NaN    NaN    25    100    115 0.0935084      1.02574 posterior_misfit           fault2-best-surp300-g25-s100/inversion_results_assim_115.pkl        ok        NaN   NaN
    4                  fault2-mean8-g25-s100                         base            8           NaN              NaN    NaN    25    100    115 0.0946213      1.03795 posterior_misfit                  fault2-mean8-g25-s100/inversion_results_assim_115.pkl        ok        NaN   NaN
    5           fault2-best-taper80-g25-s100                         best          NaN           NaN              NaN     80    25    100    115 0.0984761      1.08024 posterior_misfit           fault2-best-taper80-g25-s100/inversion_results_assim_115.pkl        ok        NaN   NaN

 rank                                    run                   experiment  mean_factor  ahead_factor  surprise_factor  taper  gate  start  assim    metric  rel_to_best      metric_path                                                                   file    status  memory_ft  seed
    1                     fault-all-g25-s100                    fault-all          NaN           NaN              NaN    NaN    25    100    120 0.0888298            1 posterior_misfit                     fault-all-g25-s100/inversion_results_assim_120.pkl        ok        NaN   NaN
    2           fault2-best-surp300-g25-s100                 best-surp300          NaN           NaN              NaN    NaN    25    100    120  0.103851      1.16911 posterior_misfit           fault2-best-surp300-g25-s100/inversion_results_assim_120.pkl        ok        NaN   NaN
    3 fault2-strong-taper80-surp300-g25-s100               strong-surp300          NaN           NaN              NaN     80    25    100    120  0.107006      1.20462 posterior_misfit fault2-strong-taper80-surp300-g25-s100/inversion_results_assim_120.pkl        ok        NaN   NaN
    4                  fault2-mean2-g25-s100                         base            2           NaN              NaN    NaN    25    100    120  0.107844      1.21405 posterior_misfit                  fault2-mean2-g25-s100/inversion_results_assim_120.pkl        ok        NaN   NaN
    5   fault2-best-taper80-surp300-g25-s100                 best-surp300          NaN           NaN              NaN     80    25    100    120  0.111592      1.25625 posterior_misfit   fault2-best-taper80-surp300-g25-s100/inversion_results_assim_120.pkl        ok        NaN   NaN


rank                                    run                   experiment  mean_factor  ahead_factor  surprise_factor  taper  gate  start  assim   metric  rel_to_best      metric_path                                                                   file    status  memory_ft  seed
    1         fault2-strong-taper80-g25-s100                       strong          NaN           NaN              NaN     80    25    100    125 0.123309            1 posterior_misfit         fault2-strong-taper80-g25-s100/inversion_results_assim_125.pkl        ok        NaN   NaN
    2 fault2-strong-taper80-surp300-g25-s100               strong-surp300          NaN           NaN              NaN     80    25    100    125 0.124249      1.00762 posterior_misfit fault2-strong-taper80-surp300-g25-s100/inversion_results_assim_125.pkl        ok        NaN   NaN
    3           fault2-best-surp300-g25-s100                 best-surp300          NaN           NaN              NaN    NaN    25    100    125 0.129665      1.05154 posterior_misfit           fault2-best-surp300-g25-s100/inversion_results_assim_125.pkl        ok        NaN   NaN
    4   fault2-best-taper80-surp300-g25-s100                 best-surp300          NaN           NaN              NaN     80    25    100    125 0.133627      1.08368 posterior_misfit   fault2-best-taper80-surp300-g25-s100/inversion_results_assim_125.pkl        ok        NaN   NaN
    5           fault2-best-taper80-g25-s100                         best          NaN           NaN              NaN     80    25    100    125 0.135695      1.10045 posterior_misfit           fault2-best-taper80-g25-s100/inversion_results_assim_125.pkl        ok        NaN   NaN
  rank                                    run                   experiment  mean_factor  ahead_factor  surprise_factor  taper  gate  start  assim    metric  rel_to_best      metric_path                                                                   file    status  memory_ft  seed
    1           fault2-best-surp300-g25-s100                 best-surp300          NaN           NaN              NaN    NaN    25    100    130 0.0673205            1 posterior_misfit           fault2-best-surp300-g25-s100/inversion_results_assim_130.pkl        ok        NaN   NaN
    2 fault2-strong-taper80-surp300-g25-s100               strong-surp300          NaN           NaN              NaN     80    25    100    130 0.0691929      1.02781 posterior_misfit fault2-strong-taper80-surp300-g25-s100/inversion_results_assim_130.pkl        ok        NaN   NaN
    3         fault2-strong-taper80-g25-s100                       strong          NaN           NaN              NaN     80    25    100    130 0.0701314      1.04175 posterior_misfit         fault2-strong-taper80-g25-s100/inversion_results_assim_130.pkl        ok        NaN   NaN
    4           fault2-best-taper80-g25-s100                         best          NaN           NaN              NaN     80    25    100    130 0.0727675      1.08091 posterior_misfit           fault2-best-taper80-g25-s100/inversion_results_assim_130.pkl        ok        NaN   NaN
    5                  fault2-mean8-g25-s100                         base            8           NaN              NaN    NaN    25    100    130  0.073372      1.08989 posterior_misfit                  fault2-mean8-g25-s100/inversion_results_assim_130.pkl        ok        NaN   NaN
    rank                                    run                   experiment  mean_factor  ahead_factor  surprise_factor  taper  gate  start  assim   metric  rel_to_best      metric_path                                                                   file    status  memory_ft  seed
    1                       hpc-full-memory1             hpc-full-memory1          NaN           NaN              NaN    NaN   NaN    NaN    140 0.123922            1 posterior_misfit                       hpc-full-memory1/inversion_results_assim_140.pkl        ok        NaN   NaN
    2          fault2-best-taper150-g25-s100                         best          NaN           NaN              NaN    150    25    100    140 0.168754      1.36178 posterior_misfit          fault2-best-taper150-g25-s100/inversion_results_assim_140.pkl        ok        NaN   NaN
    3            fault2-ft100-mean2-g25-s100                         base            2           NaN              NaN    NaN    25    100    140 0.175908      1.41951 posterior_misfit            fault2-ft100-mean2-g25-s100/inversion_results_assim_140.pkl        ok        100   NaN
    4                     fault-all-g25-s100                    fault-all          NaN           NaN              NaN    NaN    25    100    140 0.181853      1.46748 posterior_misfit                     fault-all-g25-s100/inversion_results_assim_140.pkl        ok        NaN   NaN
    5           fault2-best-taper80-g25-s100                         best          NaN           NaN              NaN     80    25    100    140 0.189406      1.52844 posterior_misfit           fault2-best-taper80-g25-s100/inversion_results_assim_140.pkl        ok        NaN   NaN
    6                    fault-mean-g25-s100                   fault-mean          NaN           NaN              NaN    NaN    25    100    140 0.190298      1.53563 posterior_misfit                    fault-mean-g25-s100/inversion_results_assim_140.pkl        ok        NaN   NaN
    7   fault2-best-taper80-surp300-g25-s100                 best-surp300          NaN           NaN              NaN     80    25    100    140 0.193368      1.56041 posterior_misfit   fault2-best-taper80-surp300-g25-s100/inversion_results_assim_140.pkl        ok        NaN   NaN
  