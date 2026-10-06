# 2D HPC validation — 2026-10-02

## TV implementation and bounded pilot — 2026-10-05

`--regularization tv --single-update` enables member-wise 1D TV proximal steps
and a spacing-aware fixed-Jacobian 2D IRLS solve with weak independent amplitude
regularization. Hard mode projection is disabled. `--tv-weight` is the tracked
total spatial dose per point; `--tv-iterations` caps local numerical work, not
simulator calls. Checkpoint metadata distinguishes Gaussian diagnostic losses,
TV settings/history and linearized versus true-forward predictions.

All 47 tests pass. Bounded pilots are under
`runs/tv-pilot-20261005T111823Z/`; see `TV_PILOT_ASSESSMENT.md` for design,
independent forward checks and limitations. TV 0.2 improves both 1D recovery
and verified fit in the short unforced zones. Forced 2D remains mixed: verified
misfit 60.52 / 45.50 / 90.80 for TV 0 / 0.05 / 0.2, versus optimistic linearized
4.90 / 4.26 / 2.94. The forced points take 92–96 s, exceeding the minute target.
No full rerun was launched; the pilot does not approve a global TV strength.

## Corrected prior units and matched projection check — 2026-10-05

Internal resistivity coordinates are **natural logs**, as confirmed by
`np.exp` in both simulator paths. Earlier claims that these coordinates were
log10, and RMSE comparisons made between the two log bases, are superseded by
`PROJECTION_PILOT_ASSESSMENT.md`. The blocky CLI still accepts log10 levels
and width; `prior_parameters_ln` converts them once. Defaults now correspond
exactly to 1.5/30 Ohm m; manifests and checkpoints record physical values,
natural-log values, width and radius. `--prior-projection on/off` supports a
matched comparison with unchanged chaining.

Single-update checkpoints explicitly label predictions and misfits
`linearized`, store the independently evaluated prior misfit separately, and
record per-point time. `plot_curtain.py` distinguishes linearized estimates
from forward-evaluated posteriors, including older single-update checkpoints.
`check_posterior.py` performs an independent forward-only posterior batch,
recording the original covariance, saved mask and checkpoint checksum.

Bounded matched pilots (three points per zone) completed under
`runs/projection-corrected-20261005T103027Z`. Projection improved transition
RMSE (0.209→0.167 log10) but worsened reservoir RMSE (0.382→0.776 log10).
The one 2D point took 92.9 s, exceeding the one-minute goal. Its independent
posterior check succeeded for all five members: true mean misfit 95.31 versus
linearized 91.40. No full inversion was relaunched. See the assessment for
scope limits and acceptance decision.

## r2 terminated as unified-domain baseline — 2026-10-06 08:0x UTC

Run `benchmark3-full-20261005-r2` terminated by user decision at index 130
(130 checkpoints, indices 0–129; supervisor recorded FAILED/terminated).
Final breakdown: 55 pure-1D points, 52 successful 2D single-update
engagements, 23 abandoned→1D fallbacks (cap 8), 220 substitutions total.
52 engaged indices: 31–129 (dense through the 31–93 hard zone). This
partial run is the agreed comparison baseline for the unified-domain
implementation (branch `2D-curtain`); the code state at termination is
frozen on branch `TV-prior`. The known 6h42m local-stall failure mode
(kriging/`fsolve` in `resample_prior_multiple_1D_to_2D` at degenerate
ensembles) motivated the unified redesign: one moving 2D window state,
sample-correlation ES updates at all fidelities, conditional blocky edge
entry, exit = drop/finalize.

## Index-32 stall post-mortem and substitution cap — 2026-10-05 (evening)

The full TV rerun `benchmark3-full-20261005` stalled ~3 h 59 m at index 32 and
then crashed with `MemberEvaluationError` after exhausting **all 245 remaining
ensemble members** (245 logged substitutions, each costing two ~25–60 s remote
attempts; exit-174 SIGSEGV class on every candidate, on multiple nodes).

Root cause (two-sided):

1. **Pathological prior construction at index 32.** Index 31's 2D posterior
   slots 0–2 sit pinned at the upper Rh clamp (ln 4.0 = 54.6 Ohm m) with
   ~4.3 ln-unit lateral contrasts per cell; index 32 pairs slot 0's aligned
   flanks (carried unchanged through every substitution,
   `substitute_prior()` in `run_DA_sequential.py`) with its own degenerate
   1D ensemble (1D misfit 1131 → 289). Every resulting 2D field crashes
   fwd2D regardless of member or node. A root-cause fix (flank hygiene at
   engagement) is future work; not addressed in this change.
2. **No substitution bound.** `substitute()` looped until ensemble
   exhaustion; `_initialize_members` then raised `MemberEvaluationError`
   and killed the controller — a single hard point could burn hours.

Fix: `--max-substitutions` (default 8, validated ≥ 0) caps substitutions per
engaged point. On cap (or exhaustion) the controller catches
`MemberEvaluationError`, restores the 1D prior mean/covariance and TV
history, and completes the point through the ordinary final-1D update:
checkpoint records `selected_fidelity='1D'`, `two_d_mode='abandoned'`,
`escalation_abandoned`, the real `solver_failures`/`substituted_members`
and burned `two_d_calls`; chaining continues from the 1D posterior.
Verified by `test_controller_abandons_2d_after_substitution_cap_and_falls_back_to_1d`
(50 tests pass).

Relaunched as `runs/benchmark3-full-20261005-r2/` (same recommended TV
config plus `--max-substitutions 8`; status `…-r2.status.json`, log
`…-r2.log`). Verified on the hard stretch: index 31 engaged cleanly
(5 calls, 154 s); index 32 hit the cap (8 substitutions, 22 calls, 12 min)
and completed as a provenance-tagged 1D point (misfit 289 → 234, TV dose
bookkeeping exact); index 33 engaged 2D successfully **after** 5
substitutions (misfit 647 → 173) — the cap bounds cost without abandoning
salvageable points.

## Full TV rerun launched — 2026-10-05 13:05 UTC

Artifact directory: `runs/benchmark3-full-20261005/`. All 281 points, seed
2026, 250 ES-MDA members, 5 optimized members, threshold 100 (user policy),
strict chaining, blocky prior, projection off, single-update 2D, and the
recommended TV configuration from `TV_PILOT_ASSESSMENT.md`:
`--regularization tv --tv-weight-1d 0.1 --tv-weight-2d 0.05
--tv-max-step 0.25`. Backend: slurm/comp/energytech, 1 rank, 5 cpus,
16 GiB, `--walltime 00:05:00` (node-variance headroom; per-evaluation
timeout 180 s still binds first for hung members), `--poll-max-delay 5`,
binary `/cluster/home/krfo/adjoint-2d-v1`, tolerance 1e-6.

Detached launch via `supervise_run.py` (status:
`runs/benchmark3-full-20261005.status.json`, log:
`runs/benchmark3-full-20261005.log`); curtain follows via a detached
`plot_curtain.py --follow` (`…curtain.log`). Early indices are 1D in
seconds; engaged 2D points use one five-member Jacobian batch each
(~1.5–2 min under load). First observed engagement: index 31 (1D final
misfit ≈ 163 > 100).

## Polling cap, node variance and wave scheduling — 2026-10-05

Forced-2D diagnostics showed batch wait ≈ 75 s for 44–68 s member solves.
Receipts (`jobs/*/status.json`, `timings.jsonl`) attribute the remainder to
client polling: `evaluate_batch` backed off up to 30 s between polls.
`metadata['poll_max_delay_seconds']` (set by `esmda_remote2d` from
`DA_SIM_POLL_MAX_DELAY` / `--poll-max-delay`, default 5 s, minimum 1 s)
now caps the backoff; detection latency is bounded by the cap and batch
`wait_seconds` dropped to solve time + O(5 s). Verified by
`tests/test_polling.py` against a fake backend.

Per-member Jacobian solve time varies ~2× by placement (node-02 39–53 s,
node-07 55–68 s for identical fields; one busy-period wave measured
58–105 s across five nodes). Pilot array-job walltime is therefore
`00:02:00` (`pilot_tv2.py --walltime`); a 60 s limit killed members by
placement, not by physics (`sacct` State TIMEOUT on slow nodes). Batched
diagnostics must also run **one run at a time**: four concurrent forced
runs (20 simultaneous members) oversubscribe the node and push solves
past any reasonable walltime — the first stage-TV wave lost 3 of 4 runs
to exactly this. `pilot_tv2.py` schedules forced runs and forward checks
sequentially and records per-batch node placement plus member solve
seconds in `assessment.json`.

## collect() alignment fix and single-update mode — 2026-10-05

The full run `benchmark3-full-20261004` failed at index 36 when a trial
batch lost 3 of 5 fields (2 solver crashes + 1 timeout — the absorbed
failure class). The killer was a placeholder bug in the Dashboard's
`collect()`: for FAILED fields the (0,0) full-Jacobian placeholder was
appended only `if full_jacobians:` — empty when field 0 fails first —
shifting every subsequent Jacobian by one field. Field 1 silently received
field 2's Jacobian and field 2 (finite prediction) paired with the (0,0)
placeholder, raising `ValueError` and killing the controller. Audit: the
crashed batch was the only misaligned batch across all multi-field runs —
no silent corruption occurred. Fixes:

1. `Dashboard/src/pxwf/execution/backend.py`: `wants_full` is read from the
   batch metadata before the field loop and drives the FAILED-field
   placeholder in both `SlurmBackend.collect()` and `LocalBackend`'s inline
   collection — alignment is now guaranteed by construction.
2. `esmda_remote2d.run_grad_multi`: a length-mismatched collect degrades
   every field to a failure entry (rejected trial, damping retry) instead of
   raising; per-field empty Jacobians map to that field failing. Only a
   worker without full-Jacobian support still raises.
3. Regression tests reproduce the misaligned-collect crash scenario.

**Single-update mode** (`--single-update`, user decision 2026-10-05: one
batch per engaged point, threshold ~30, strict sequential chaining):
engaged 2D points evaluate the member priors in one multi-field batch,
apply one damped linearized (ES-MDA-style) update per member, and report
posterior predictions through `f(x) + J dx` — no solver trials. The
linearized loss is not monotone (Bayes update, not optimization); the next
point re-linearizes around the moved prior. `update_members_once` in
`esmda_core.py` shares the initial-evaluation/substitution phase with
`optimize_members`. Verified at index 80 (forced): 5 field-evals, ~68 s
2D stage, 128 s point total, posterior misfits 160–211 from a 1D misfit
of 184. Checkpoints record `two_d_mode`.

## 2D acceleration and fidelity research — 2026-10-04

User concerns: 2D escalation is extremely slow (should fail fast), the 1D→2D
data misfit jumps, and the Jacobian/total derivative needs verification.

**Jacobian (validated).** Production-scale FD validation (8 tools, 15×5×20
window, trajectory index 27; `runs/validate2d-*-20261004`): adjoint-vs-FD
relative L2 sits at or below the FD self-consistency floor (fd-vs-fd) in 7/8
checks at steps 0.03/0.01. The floor itself is fwd2D convergence noise (error
bound stalls ~2e-4 mid-solve before converging ~5e-7) plus nonlinearity;
steps ≤0.001 are noise-dominated and unusable at this scale. LAS outputs
carry 8 significant digits (not the limiter). Conclusion: the full
optimizer-coordinate total derivative — including the ∂lnRv/∂lnRh chain-rule
term and the latent block — is consistent with the forward within FD
attainable precision.

**1D↔2D consistency (good).** For x-invariant media (homogeneous; mild and
factor-10 rh steps; latent steps; the actual index-27 1D posterior column)
UTA1D and fwd2D agree within 0.3–2.9% per tool. Two bugs were found on the
way: the controller fed UTA1D `surface_depth` at cell *centers* while EMsim
interprets `surface_depth[1:]` as layer boundaries (fixed to edges, `i*dZ`),
and validate_2d's posterior base built a transposed tile (fixed). The
fidelity gap on representable channels is therefore ~2–3%, not the driver of
the misfit jumps.

**Real driver: unrepresentable ultra-shallow channels.** The USDP/USDA
components carry near-zero data with signal-scaled σ of 0.0016–0.008 — up to
30× smaller than the smallest 2D-representable channel (0.059) — while the 2D
grid (5 ft cells) produces 0.3–1.2-unit sub-cell artifacts there, worth
reduced misfits of 10³–2·10⁴ per channel. These 16 of 64 channels dominated
the entire 2D loss (and half the residual 1D misfit). They are now masked
globally (`masked_udar_components = ('USDP','USDA')`), unifying the earlier
noise-floor masking (that population was the same channels at early indices
where σ collapses below 1e-3).

**Prior propagation.** The 2D prior conditioned only columns 0–7 and re-drew
the active right half from the full geostatistical prior (member std 0.50 at
column 12 vs 0.02 at the bit; deviations to 1.8 log10 units). The residual
covariance (and the LM's C_theta) is now scaled by the lower-fidelity
posterior spread (`posterior_spread`, clip [0.03, 1]× of the prior std):
cold starts use the final 1D ensemble spread, warm starts the aligned
posterior spread.

**Fail fast.** `run_grad_multi` submits one multi-field array job per LM step
(count=5, `--concurrency 5`, `tolerate_failures` for per-field NaN handling);
`optimize_members` steps all members per iteration with per-member damping,
substitution on double initial failure, stop on 2 consecutive small (<5%)
improvements or 2 consecutive rejects; `evaluation_timeout_seconds`
600→180. Result at index 27 (forced 2D, 250 members, 5 optimized): initial
member losses 527–2230 (was 2·10⁷ in the crashed run), posterior misfits
45–171 (was ~6000–15000), 2–4 iterations, ~5 min per 2D point (was ~26 min).
With US masking many points now pass the 1D gate (index 27: 3.63 < 4).

## Crash post-mortem and robustness fixes — 2026-10-03

The full run `benchmark3-full-20261002` failed 43 minutes in at index 2,
2D member 2: `fwd2D` died with `forrtl: severe (174): SIGSEGV` (confirmed in
the batch work dir `solver.log` on the cluster) while evaluating its 2D
**prior**. That input was the only one of 285 batches with max log10 Rh
4.99 (all successful batches stayed ≤ 3.96): the 1D ES-MDA updates against
noise-floor channels had pushed a selected member to ≈10⁵ Ω·m, and the
solver stalled ~2400 iterations before segfaulting. The exception escaped
through the unprotected *initial* evaluation in `optimize_member`
(esmda_core.py) and killed the controller.

Three changes (user-approved 2026-10-03):

1. **log10 Rh clamped to `[-1, 4]`** (`LOG_RH_LIMITS`, esmda_core.py)
   after every prior sample, every ES-MDA ensemble update, the 2D prior
   resample and inside every LM trial. Posterior fields now provably stay
   inside the range every successful batch occupied.
2. **Noise-floor channel masking**: any channel with noise standard
   deviation below 1e-3 (`noise_floor_mask`) is excluded from the misfit,
   ES-MDA updates, loss and LM rows. Dataset check: exactly 400 such
   channels (16 per index, early trajectory; none past ~index 75) with
   sigma ≤ 9.1e-4, while all other channels have sigma ≥ 1.3e-3. A
   datum-based test (|d| ≤ 3σ) was rejected: 103 floor channels have
   |d| > 3σ and would stay toxic. Checkpoints carry `observation_mask`
   and `masked_channels`; stored observations/covariances/predictions
   are masked-space.
3. **Retry + member substitution**: the initial 2D evaluation is retried
   once; if it still fails, `MemberEvaluationError` triggers substitution
   of the next-ranked ensemble member (prior rebuilt for that slot),
   recorded in `solver_failures`, `member_candidates` and
   `substituted_members`. The run aborts only if the whole ensemble is
   exhausted.

Regression suite grew from 18 to 23 tests (clamp, noise-floor mask,
initial-evaluation retry, controller-level substitution and masking).

Pilot `benchmark3-fullpilot-20261003` (indices 0–5, 250 members, 2
optimized, unforced) completed in ~90 seconds: each index masks 16
channels, 1D suffices everywhere (final misfit 1.2–2.3 < threshold 4),
zero 2D calls, MDA weight exactly 1.0, posterior log10 Rh within
[-0.09, 0.95]. The early trajectory no longer escalates; indices ~75+
(no floor channels, 1D misfit ~240–350) still escalate to 2D.

## Adaptive fidelity selection — 2026-10-02

The controller now gates 2D engagement on the measurement-only reduced
misfit `mean((f_ens_mean - d)^2 / Cd)` against the original covariance
(`--misfit-threshold`, default 4). The same criterion stops the 0D/1D
ES-MDA loops. When 1D is sufficient, one final 1D ES-MDA update consumes
the remaining inflation weight `1/(1-w)` and no 2D solves are submitted; a
misfit increase from that final update is reported
(`final_iteration_worsened`) and the state is kept. After a 1D-only point
the next 2D prior is built cold from the 1D columns. Checkpoints carry the
audit fields `selected_fidelity`, `misfit`, `misfit_threshold`,
`posterior_misfit`, `two_d_calls`, `mda_weight`, `lower_fidelity_weight`,
`final_inflation_factor`, `final_iteration_worsened`. The regression suite
grew from 14 to 18 tests covering the gate, the final update,
worsened-misfit reporting and cold-start priors. `plot_curtain.py` renders
the growing inversion curtain (`--follow` mode refreshes on new
checkpoints).

On the observed Benchmark-3 data the 1D reduced misfit after the full
lower-fidelity budget is on the order of 300–400, far above the default
threshold 4: every logged point escalates to 2D. The threshold is
configurable if 1D-only points are desired.

## Reused runtime

The controller runs in `../Dashboard/.venv` (Python 3.12.3; NumPy 2.5.3,
SciPy 1.18.1, pandas 3.0.6, h5py 3.16.0). No environment installation or
solver rebuild was needed.

| Setting | Verified value |
|---|---|
| SSH alias | `HPC` |
| Login host | `login-1.hpc.intra.norceresearch.no` |
| Slurm partition/account/QoS | `comp` / `energytech` / `normal` |
| Resources per validation/pilot array element | 1 node, 1 rank, 5 CPUs, 16 GiB, 20 minutes |
| Remote Python | `/cluster/home/krfo/pxwf-runtime/.venv/bin/python` |
| Remote packages | NumPy 2.5.3, h5py 3.16.0 |
| Pinned executable | `/cluster/home/krfo/adjoint-2d-v1/fwd2D` |
| OneAPI | `/cluster/home/krfo/intel/oneapi/2026.1/oneapi-vars.sh` |
| HDFql | `/cluster/home/krfo/adjoint-2d-v1` |
| MPI plugin | `pmi2` |
| Adjoint spec SHA-256 | `fa45a832ed8deb1303d606d557282b4a7c2ec09fb3e53537ca72c76446ce737f` |

Each batch records the deployed worker hash, resource request and resolved
runtime identity. The worker checks binary size/mtime and adjoint-spec hash
before execution. The v2 solver exists but was not used in these validations.

## Gradient validation

Artifact: `runs/hpc-2d-validation-final-20261002/report.json`.

The seeded heterogeneous 8 × 12 local model has 192 optimizer parameters
and one tool (`6kHz`, `83ft`). It checks the complete prediction/gradient
path in optimizer coordinates, including spatial reordering and the
anisotropy-latent transformation. All eight directional checks and four FD
step-consistency checks passed the unchanged 2% relative-L2 limit.

| Parameter / perturbation | h=0.003 | h=0.001 |
|---|---:|---:|
| ln(Rh), single cell | 0.471% | 1.080% |
| ln(Rh), random direction | 1.156% | 0.139% |
| Anisotropy latent, single cell | 0.257% | 1.074% |
| Anisotropy latent, random direction | 0.115% | 0.690% |

Maximum FD step-to-step disagreement: **1.285%**. Forward predictions with
and without Jacobian output also agreed. All 18 final validation batches
reached `READY`.

The initial diagnostic sweeps are retained:

- `runs/hpc-2d-validation-20261002/report.json`: two anisotropy checks exceeded
  2% at h=0.0003, while every h=0.001 check passed.
- `runs/hpc-2d-validation-resolved-20261002/report.json`: ln(Rh) checks
  exceeded 2% at h=0.01; every h=0.003 check passed.

These sweeps bracket a useful FD interval. Small-step errors increased with
decreasing perturbation, consistent with solver/output precision effects;
large-step errors exposed nonlinearity. The final two-step check verifies
FD consistency as well as agreement with AD. The failed sweep reports were
not overwritten or reclassified as passing.

## Benchmark-3 pilot

Artifact directory: `runs/benchmark3-hpc-smoke-20261002/`.

Configuration: local window `(15, 5, 20)`, 5-foot cells, eight tools/64 UDAR
observations per point, logging indices 80–82, seed 2026, eight ES-MDA
members, one optimized member, one LM iteration/trial, forced 2D,
solver tolerance 1e-6.

| Logging index | Window movement (x,z) | Initial 2D member loss | Accepted loss |
|---|---|---:|---:|
| 80 | (0,0) | 7636.43 | 5183.60 |
| 81 | (1,0) | 11724.69 | 9302.65 |
| 82 | (1,0) | 9081.92 | 6275.22 |

All six HPC batches reached `READY`, with no trial errors. Batch job IDs:
`1728567`, `1728570`, `1728573`, `1728577`, `1728580`, `1728584`.
Total recorded backend time: approximately **245 seconds**.

Checkpoint verification confirmed finite parameters/predictions, shapes
`(1,600)` and `(1,64)`, correct previous-index links, 5-foot x shifts with
fixed z origin, and saved losses matching recomputation from the saved
accepted state and covariance. No jobs remained in the user's queue after
completion.

The actual pilot used generated observations in
`/tmp/opencode/esmda-data-20261002/`. The same deterministic generator was
also run into `runs/benchmark3-data-20261002/` for durable local use; each
checkpoint embeds its assimilated observations and covariance.

This is a workflow/correctness smoke test. One optimized member and one LM
iteration do not establish inversion convergence or posterior uncertainty
quality. A longer run should exercise a z-window shift on the real trajectory,
then increase ensemble/member/iteration counts before evaluating inversion
quality. The regression suite separately exercises x/z and multi-cell shifts,
negative movement and early convergence.

## Extended run launched — 2026-10-02 10:34 UTC

Artifact directory: `runs/benchmark3-hpc-extended-20261002/`.

The next sequential test covers logging indices **80–89**, with **32 ES-MDA
members**, **two optimized members**, **three LM iterations** and **two trials
per iteration**. It retains the 15 × 20 local window, 5-foot cells, eight tools,
seed 2026, forced 2D, the pinned v1 runtime and the pilot's per-task resources.
Inputs come from the durable `runs/benchmark3-data-20261002/` directory.

Preflight against the actual trajectory predicts the first z-window shift at
**index 86**, where both x and z move one cell. This is the real-trajectory
transition the three-point pilot did not reach. Completion checks should verify
posterior reuse across that transition, accepted-state/loss consistency, trial
errors, per-member loss histories and data misfit before assessing inversion
quality.

The controller is detached from the terminal and supervised. Its launch
receipt records the complete command, process IDs, start/end times and exit
status:

- Log: `runs/benchmark3-hpc-extended-20261002.log`
- Status: `runs/benchmark3-hpc-extended-20261002.status.json`
- Controller PID at launch: `119257`; supervisor PID: `119256`
- Status at launch verification: `RUNNING`; initial jobs `1728615` and
  `1728616` reached `READY` without errors, and the next job was running.

The run is **in progress**, not a completed validation. At the observed
approximately 41 seconds per batch, typical execution is on the order of an
hour; the maximum 140 evaluations would take about 97 minutes before extra
queue delays. Individual Slurm jobs have a 20-minute limit; that is not a
limit on the entire controller run.

Monitor locally:

```bash
tail -f runs/benchmark3-hpc-extended-20261002.log
../Dashboard/.venv/bin/python -B -c 'import json; s=json.load(open("runs/benchmark3-hpc-extended-20261002.status.json")); print(s["state"], "controller PID", s["controller_pid"], "exit", s.get("returncode"))'
```

The status becomes `COMPLETED` or `FAILED` when the controller exits.
Per-point checkpoints appear after both optimized members finish. Batch
requests/results and timing records remain under the run's `jobs/` directory.

To reproduce in a **fresh** directory:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
../Dashboard/.venv/bin/python -B -u run_DA_sequential.py \
  --output-dir runs/benchmark3-hpc-extended \
  --data runs/benchmark3-data-20261002/data.pkl \
  --variance runs/benchmark3-data-20261002/var.pkl \
  --sim-backend slurm --host HPC \
  --partition comp --account energytech --qos normal \
  --nodes 1 --ranks 1 --cpus 5 --memory 16 \
  --walltime 00:20:00 --concurrency 1 \
  --binary-dir /cluster/home/krfo/adjoint-2d-v1 \
  --solver-tolerance 1e-6 \
  --start 80 --stop 90 --frequency 1 \
  --ensemble-size 32 --opt-members 2 \
  --max-iterations 3 --max-lm-trials 2 --seed 2026 --force-2d
```

## Adaptive pilot — 2026-10-02 19:41 UTC

Artifact directory: `runs/benchmark3-adaptive-20261002/`. Unforced run
(indices 80–89, 32 members, 2 optimized, 3 LM iterations, threshold 4,
seed 2026). Completed with exit 0 in 94 minutes; 113/113 batches READY,
no trial errors, total MDA weight exactly 1.0 at every point, and the
z-window shift at index 86 handled. Every point escalated to 2D: the 1D
reduced misfit after the full lower-fidelity budget was 193–346, far above
the threshold 4. The 2D posterior measurement misfit improved on 1D at
most points (e.g. index 80: 345 → 288–312). The curtain plotter rendered
incrementally during the run.

## Full run launched — 2026-10-02 21:47 UTC

Artifact directory: `runs/benchmark3-full-20261002/`. All 281 logging
points, 250 ES-MDA members, 5 optimized members, 5 LM iterations, 2 trials,
threshold 4, seed 2026, pinned v1 exact-adjoint runtime, unforced. Log:
`runs/benchmark3-full-20261002.log`; status:
`runs/benchmark3-full-20261002.status.json`; curtain:
`runs/benchmark3-full-20261002/curtain.png` (refreshed every 5 minutes by
a detached `plot_curtain.py --follow` process).

Early-trajectory note: the first logging points contain structurally zero
UDAR channels (2–3 of 64) with variances near the cross-tool noise floor,
so their reduced misfits reach 1e6–1e12. The controller handles these
robustly (finite states, no trial errors), but misfits there cannot
approach the threshold; escalation is certain across the whole trajectory
at threshold 4. Estimated duration at the observed ~25–30 min per engaged
point: roughly 5–6 days.

Forced-2D check `benchmark3-2dcheck2-20261003` (index 80, 250 members, 2
optimized, 3 LM iterations) completed with exit 0: 12 2D batches all READY,
no substitutions, posterior log10 Rh within [0.33, 3.27], MDA weight exactly
1.0, posterior misfits 251/222 from a 1D misfit of 242. (An earlier
`benchmark3-2dcheck-20261003` attempt was aborted locally — the controller
process was killed by the launching shell, not by the pipeline; both its
HPC batches actually completed. Supervised launches now use
`setsid nohup ... < /dev/null` via `supervise_run.py`, which records
state/PIDs/exit in `<run>.status.json`.)

## Full run relaunched — 2026-10-03

Artifact directory: `runs/benchmark3-full-20261003/`. Same configuration as
the crashed run (all 281 points, 250 ES-MDA members, 5 optimized members,
5 LM iterations × 2 trials, threshold 4, seed 2026, pinned v1 exact-adjoint
runtime, unforced) plus the three fixes above. Log:
`runs/benchmark3-full-20261003.log`; status:
`runs/benchmark3-full-20261003.status.json`; curtain refreshed by a detached
`plot_curtain.py --follow`. Early indices run 1D-only in seconds each;
2D-escalating points cost the usual ~25–40 minutes (5 members).

## Reproduce

Use the observation-generation, gradient-validation and three-point pilot
commands in [README.md](README.md), each with a fresh output directory.
The local regression command is:

```bash
../Dashboard/.venv/bin/python -B -m pytest -q -p no:cacheprovider tests
```

Controller entry points: `run_DA_sequential.py`, `write_data.py`,
`validate_2d.py`. Shared implementation: `esmda_core.py`,
`esmda_remote2d.py`. `main(argv)` is import-safe in each entry point;
the observation generator exposes `generate_data(...)` and `read_las(...)`.
