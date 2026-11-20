#!/bin/bash
# Heavy-tailed (Student-t) increments in the 1D update, across the 600 ft transition (points 100-140).
# All runs: 1D model-error term (references/model-error-benchmark3-1d.json), gate on the 1D stage
# covariance. --student-weights repeats the 1D EnIF update of every point with a reweighted precision
# (increment_prior.py); the gate statistic is always that of the Matern round, so the same kind of point
# goes to 2D with and without reweighting.
#   matern  : no reweighting (fault5 me-1d with the gate threshold of this study)
#   members : weights at the member average of the squared increments
#   mean    : weights at the increments of the ensemble mean
# Gate threshold 1.5 (STAGE_GATE), as in fault5: "matern" then reproduces fault5-me-1d-sg1.5-s100, which is
# the reference for the reweighted runs. (A threshold of 1.0 separates the layered stretch from the fault
# in a chain that arrives from point 60, but a chain started at 100 fits its first points worse and sends
# them all to 2D, which leaves no 1D points before the fault to test the reweighting on.)
# Memory: spread 50 ft, mean 200 ft. Usage: ./run_fault_study6.sh EXPERIMENT   (or "list")
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../venv-pet-latest/bin/python}
START=${START:-100}; STOP=${STOP:-141}; SEED=${SEED:-2031}; ENS=${ENS:-250}
STAGE_GATE=${STAGE_GATE:-1.5}
case "${1:?experiment}" in
  list)    echo "matern members mean"; exit 0 ;;
  matern)  EXTRA="" ;;
  members) EXTRA="--student-weights members" ;;
  mean)    EXTRA="--student-weights mean" ;;
  *) echo "unknown experiment (try: list)" >&2; exit 2 ;;
esac
NAME=fault6-$1-sg$STAGE_GATE-s$START
[ "$SEED" = 2031 ] || NAME=$NAME-seed$SEED
[ "$ENS" = 250 ] || NAME=$NAME-ens$ENS
$PY -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
  --output-dir runs/$NAME --start $START --stop $STOP --ensemble-size $ENS --seed $SEED --mda-steps 10 \
  --tempering linear-inflation --anisotropy log_clipped --anisotropy-mean 0.5 --anisotropy-std 0.5 \
  --memory-ft 50 --model-error references/model-error-benchmark3-1d.json \
  --gate-threshold $STAGE_GATE --gate-covariance stage $EXTRA --on-gate stop --analysis enif \
  --two-d-prior ensemble_precision --directional-sigma-floor 0.5 0.25
$PY -B -u run_chain_laplace_loop.py --chain runs/$NAME --stop $STOP \
  --source references/map-source --label $NAME --map-starts 1 \
  --optimizer-setting stagnation_tolerance=0.01 --optimizer-setting stagnation_iterations=1 \
  --optimizer-setting max_rounds=2 \
  --binary-dir /cluster/home/mlie/cluster-build --orientation bedding \
  --host HPC --hpc-root /cluster/work/users/mlie/esmda-runs \
  --hpc-python /cluster/home/mlie/esmda-runs/.venv/bin/python
