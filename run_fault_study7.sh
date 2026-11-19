#!/bin/bash
# Merged logging points in the 2D stage, across the 600 ft transition (points 100-140), against
# fault4-control (same settings: no model-error term, measurement-covariance gate at 25, memory 50 ft).
# At every 2D point the problem also holds the data of the points BEHIND logging indices back (5 ft per
# index), with the tool simulated at their positions inside the current window; every point enters with
# its covariance divided by its share, and a datum's shares over its uses sum to one.
#   merge3 : current point 1/2, points 25 and 50 ft behind 1/4 each   (BEHIND="5 10", SHARES="0.5 0.25 0.25")
#   merge2 : current point 1/2, point 50 ft behind 1/2                (BEHIND="10",   SHARES="0.5 0.5")
# The 1D stage is unchanged (current point only). The tool positions of a 2D solve run side by side, in an
# allocation with as many times the solver tasks (three positions: 384 CPUs per start, 11-12 min at 115).
# Usage: ./run_fault_study7.sh EXPERIMENT      (or "list")
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../venv-pet-latest/bin/python}
# Several chains share this workstation: without a thread limit their local steps (prior fit, sampling)
# each take every core and stall one another (three at once: minutes instead of seconds).
export OPENBLAS_NUM_THREADS=${THREADS:-8} MKL_NUM_THREADS=${THREADS:-8} OMP_NUM_THREADS=${THREADS:-8}
START=${START:-100}; STOP=${STOP:-141}; SEED=${SEED:-2031}; ENS=${ENS:-250}; GATE=${GATE:-25}
# Run-time bounds of a 2D solve: MAP_STARTS extra starts from carried members besides the start at the carried
# mean (0 = that start only: half the CPUs, and the chain does not wait for the slower start), MAX_ITER caps
# the iterations (empty = no cap).
MAP_STARTS=${MAP_STARTS:-1}; MAX_ITER=${MAX_ITER:-}
# OWN_HISTORY=1: merge only once all merged points lie in this chain (the first points use the bit data alone).
OWN_HISTORY=${OWN_HISTORY:-}
case "${1:?experiment}" in
  list)   echo "merge3 merge2 control"; exit 0 ;;
  control) BEHIND=""; SHARES="" ;;      # same chain without merging (reference for other START or run-time bounds)
  merge3) BEHIND="5 10"; SHARES="0.5 0.25 0.25" ;;
  merge2) BEHIND="10"; SHARES="0.5 0.5" ;;
  *) echo "unknown experiment (try: list)" >&2; exit 2 ;;
esac
OWN_FLAG=""; [ -z "$OWN_HISTORY" ] || [ -z "$BEHIND" ] || OWN_FLAG="--merge-own-history"
NAME=fault7-$1-g$GATE-s$START
[ "$SEED" = 2031 ] || NAME=$NAME-seed$SEED
[ "$ENS" = 250 ] || NAME=$NAME-ens$ENS
[ "$MAP_STARTS" = 1 ] || NAME=$NAME-starts$MAP_STARTS
[ -z "$MAX_ITER" ] || NAME=$NAME-it$MAX_ITER
[ -z "$OWN_FLAG" ] || NAME=$NAME-own
# An existing chain is continued (e.g. with a later STOP) by the loop below; only a new chain is started here.
[ -f runs/$NAME/study.json ] || \
$PY -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
  --output-dir runs/$NAME --start $START --stop $STOP --ensemble-size $ENS --seed $SEED --mda-steps 10 \
  --tempering linear-inflation --anisotropy log_clipped --anisotropy-mean 0.5 --anisotropy-std 0.5 \
  --memory-ft 50 --gate-threshold $GATE ${BEHIND:+--merge-behind $BEHIND --merge-shares $SHARES} $OWN_FLAG \
  --on-gate stop --analysis enif \
  --two-d-prior ensemble_precision --directional-sigma-floor 0.5 0.25
$PY -B -u run_chain_laplace_loop.py --chain runs/$NAME --stop $STOP \
  --source references/map-source --label $NAME --map-starts $MAP_STARTS \
  --optimizer-setting stagnation_tolerance=0.01 --optimizer-setting stagnation_iterations=1 \
  --optimizer-setting max_rounds=2 ${MAX_ITER:+--optimizer-setting max_iterations=$MAX_ITER} \
  --binary-dir /cluster/home/mlie/cluster-build --orientation bedding \
  --host HPC --hpc-root /cluster/work/users/mlie/esmda-runs \
  --hpc-python /cluster/home/mlie/esmda-runs/.venv/bin/python
