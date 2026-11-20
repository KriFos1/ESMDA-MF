#!/bin/bash
# Model-error study across the 600 ft transition (logging points 100-140), against fault4-control.
# The data covariance so far holds only the 2.5 % measurement term, while the true model misfits the data
# by 4-30 measurement std in the calm layered stretch (1D: structure the 1D model cannot represent; 2D:
# faults outside the window and the 5 ft grid). --model-error adds a per-channel term to the covariance
# each stage assimilates with: Cd (1 + factor^2), factors from the benchmark truth residuals
# (build_model_error.py; references/model-error-benchmark3*.json). Misfits keep the measurement Cd.
#   me-2d : term in the 2D stage only; the gate is unchanged, so the same points go to 2D as in the control
#   me    : term in both stages; the gate statistic uses the 1D stage covariance (--gate-covariance stage),
#           so it opens when the 1D misfit exceeds what the known 1D model error explains
#   me-1d : term in the 1D stage only, same gate as "me"
# Memory: spread 50 ft, mean 200 ft (the driver default mean factor 4).
# Usage: ./run_fault_study5.sh EXPERIMENT      (EXPERIMENT: see the case list, or "list")
#   STAGE_GATE=2 ./run_fault_study5.sh me      (threshold of the stage-covariance gate, default 1.5: in a
#     1D-only pass the statistic is 0.4-0.9 in the calm stretch 69-102 and 2-22 inside the fault 106-131)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../venv-pet-latest/bin/python}
START=${START:-100}; STOP=${STOP:-141}; SEED=${SEED:-2031}; ENS=${ENS:-250}
GATE=${GATE:-25}; STAGE_GATE=${STAGE_GATE:-1.5}
TABLE=references/model-error-benchmark3
case "${1:?experiment}" in
  list)   echo "me-2d me me-1d"; exit 0 ;;
  me-2d)  EXTRA="--model-error $TABLE-2d.json --gate-threshold $GATE"; TAG=g$GATE ;;
  me)     EXTRA="--model-error $TABLE.json --gate-threshold $STAGE_GATE --gate-covariance stage"; TAG=sg$STAGE_GATE ;;
  me-1d)  EXTRA="--model-error $TABLE-1d.json --gate-threshold $STAGE_GATE --gate-covariance stage"; TAG=sg$STAGE_GATE ;;
  *) echo "unknown experiment (try: list)" >&2; exit 2 ;;
esac
NAME=fault5-$1-$TAG-s$START
[ "$SEED" = 2031 ] || NAME=$NAME-seed$SEED
[ "$ENS" = 250 ] || NAME=$NAME-ens$ENS
$PY -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
  --output-dir runs/$NAME --start $START --stop $STOP --ensemble-size $ENS --seed $SEED --mda-steps 10 \
  --tempering linear-inflation --anisotropy log_clipped --anisotropy-mean 0.5 --anisotropy-std 0.5 \
  --memory-ft 50 $EXTRA --on-gate stop --analysis enif \
  --two-d-prior ensemble_precision --directional-sigma-floor 0.5 0.25
$PY -B -u run_chain_laplace_loop.py --chain runs/$NAME --stop $STOP \
  --source references/map-source --label $NAME --map-starts 1 \
  --optimizer-setting stagnation_tolerance=0.01 --optimizer-setting stagnation_iterations=1 \
  --optimizer-setting max_rounds=2 \
  --binary-dir /cluster/home/mlie/cluster-build --orientation bedding \
  --host HPC --hpc-root /cluster/work/users/mlie/esmda-runs \
  --hpc-python /cluster/home/mlie/esmda-runs/.venv/bin/python
