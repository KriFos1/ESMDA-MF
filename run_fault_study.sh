#!/bin/bash
# Memory-option study around the 600 ft transition (logging points ~100-140). Usage:
#   ./run_fault_study.sh VARIANT        VARIANT: base | ahead | mean | latent | all
#   START=100 STOP=141 GATE=25 ./run_fault_study.sh all      (GATE=1.5 sends every point to 2D)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../venv-pet-latest/bin/python}
START=${START:-100}; STOP=${STOP:-141}; GATE=${GATE:-25}
case "${1:?variant}" in
  base)   EXTRA="" ;;
  ahead)  EXTRA="--memory-ahead-factor 0.5 --memory-surprise-factor 0.3" ;;
  mean)   EXTRA="--memory-mean-factor 4" ;;
  latent) EXTRA="--memory-latent-factor 3" ;;
  all)    EXTRA="--memory-ahead-factor 0.5 --memory-surprise-factor 0.3 --memory-mean-factor 4 --memory-latent-factor 3" ;;
  *) echo "unknown variant" >&2; exit 2 ;;
esac
NAME=fault-$1-g$GATE-s$START
$PY -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
  --output-dir runs/$NAME --start $START --stop $STOP --ensemble-size 250 --seed 2031 --mda-steps 10 \
  --tempering linear-inflation --anisotropy log_clipped --anisotropy-mean 0.5 --anisotropy-std 0.5 \
  --memory-ft 50 $EXTRA --gate-threshold $GATE --on-gate stop --analysis enif \
  --two-d-prior ensemble_precision --directional-sigma-floor 0.5 0.25
$PY -B -u run_chain_laplace_loop.py --chain runs/$NAME --stop $STOP \
  --source references/map-source --label $NAME --map-starts 1 \
  --optimizer-setting stagnation_tolerance=0.01 --optimizer-setting stagnation_iterations=1 \
  --optimizer-setting max_rounds=2 \
  --binary-dir /cluster/home/mlie/cluster-build --orientation bedding \
  --host HPC --hpc-root /cluster/work/users/mlie/esmda-runs \
  --hpc-python /cluster/home/mlie/esmda-runs/.venv/bin/python
