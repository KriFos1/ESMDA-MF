#!/bin/bash
# Closing memory study across the 600 ft transition (logging points 100-140). The earlier studies settled on
# spread memory 50 ft with the ensemble mean remembered for 200 ft (mean factor 4, now the driver default);
# the ahead factor, taper, latent factor and surprise are removed. This script tested whether a shorter
# SPREAD memory helps once the MEAN memory is held at 200 ft (mem25-mean8: no; mem10-mean20: not run).
# The surp0.3 run of this study (surprise from the forecast misfit) was 5 % worse than the control.
# Usage: ./run_fault_study4.sh EXPERIMENT      (EXPERIMENT: see the case list, or "list")
#   ENS=500 ./run_fault_study4.sh control      (larger ensemble; the size goes into the run name)
#   SEED=2032 ./run_fault_study4.sh control    (seed replicate; the seed goes into the run name)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../venv-pet-latest/bin/python}
START=${START:-100}; STOP=${STOP:-141}; GATE=${GATE:-25}; SEED=${SEED:-2031}; ENS=${ENS:-250}
FT=50; MEAN=4; EXTRA=""
case "${1:?experiment}" in
  list)        echo "control mem25-mean8 mem10-mean20"; exit 0 ;;
  control)     ;;                                    # spread 50 ft, mean 200 ft
  mem25-mean8) FT=25; MEAN=8 ;;                      # spread 25 ft, mean 200 ft
  mem10-mean20) FT=10; MEAN=20 ;;                    # spread 10 ft, mean 200 ft
  *) echo "unknown experiment (try: list)" >&2; exit 2 ;;
esac
NAME=fault4-$1-g$GATE-s$START
[ "$SEED" = 2031 ] || NAME=$NAME-seed$SEED
[ "$ENS" = 250 ] || NAME=$NAME-ens$ENS
$PY -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
  --output-dir runs/$NAME --start $START --stop $STOP --ensemble-size $ENS --seed $SEED --mda-steps 10 \
  --tempering linear-inflation --anisotropy log_clipped --anisotropy-mean 0.5 --anisotropy-std 0.5 \
  --memory-ft $FT --memory-mean-factor $MEAN $EXTRA --gate-threshold $GATE --on-gate stop --analysis enif \
  --two-d-prior ensemble_precision --directional-sigma-floor 0.5 0.25
$PY -B -u run_chain_laplace_loop.py --chain runs/$NAME --stop $STOP \
  --source references/map-source --label $NAME --map-starts 1 \
  --optimizer-setting stagnation_tolerance=0.01 --optimizer-setting stagnation_iterations=1 \
  --optimizer-setting max_rounds=2 \
  --binary-dir /cluster/home/mlie/cluster-build --orientation bedding \
  --host HPC --hpc-root /cluster/work/users/mlie/esmda-runs \
  --hpc-python /cluster/home/mlie/esmda-runs/.venv/bin/python
