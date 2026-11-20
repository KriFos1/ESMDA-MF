#!/bin/bash
# Follow-up to run_fault_study.sh: pending memory-mean-factor, gate and seed tests (5 ft logging spacing).
# Usage: ./run_fault_study2.sh EXPERIMENT      (EXPERIMENT: see the case list, or "list")
#   START=100 STOP=141 ./run_fault_study2.sh mean8
# "best" = ahead 0.5 / surprise 0.3 + mean factor 4 (the previous study's best without the ineffective latent factor).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../venv-pet-latest/bin/python}
START=${START:-100}; STOP=${STOP:-141}
AHEAD="--memory-ahead-factor 0.5 --memory-surprise-factor 0.3"
STRONG="--memory-ahead-factor 0.25 --memory-surprise-factor 0.1"
FT=50; GATE=25; SEED=2031; EXTRA=""
case "${1:?experiment}" in
  list)         echo "mean2 mean3 mean8 ft100-mean2 best-mean2 best-mean8 best-g15 best-g50 best-g1.5 base-s2032 base-s2033 best-s2032 best-s2033 best-taper80 best-taper150 strong-taper80 strong-taper150 best-surp300 best-taper80-surp300 strong-taper80-surp300"; exit 0 ;;
  mean2)        EXTRA="--memory-mean-factor 2" ;;
  mean3)        EXTRA="--memory-mean-factor 3" ;;
  mean8)        EXTRA="--memory-mean-factor 8" ;;
  ft100-mean2)  FT=100; EXTRA="--memory-mean-factor 2" ;;
  best-mean2)   EXTRA="$AHEAD --memory-mean-factor 2" ;;
  best-mean8)   EXTRA="$AHEAD --memory-mean-factor 8" ;;
  best-g15)     GATE=15;  EXTRA="$AHEAD --memory-mean-factor 4" ;;
  best-g50)     GATE=50;  EXTRA="$AHEAD --memory-mean-factor 4" ;;
  best-g1.5)    GATE=1.5; EXTRA="$AHEAD --memory-mean-factor 4" ;;
  base-s2032)   SEED=2032 ;;
  base-s2033)   SEED=2033 ;;
  best-s2032)   SEED=2032; EXTRA="$AHEAD --memory-mean-factor 4" ;;
  best-s2033)   SEED=2033; EXTRA="$AHEAD --memory-mean-factor 4" ;;
  best-taper80)    EXTRA="$AHEAD --memory-mean-factor 4 --memory-taper-ft 80" ;;
  best-taper150)   EXTRA="$AHEAD --memory-mean-factor 4 --memory-taper-ft 150" ;;
  strong-taper80)  EXTRA="$STRONG --memory-mean-factor 4 --memory-taper-ft 80" ;;
  strong-taper150) EXTRA="$STRONG --memory-mean-factor 4 --memory-taper-ft 150" ;;
  best-surp300)           EXTRA="$AHEAD --memory-mean-factor 4 --surprise-misfit 3 300" ;;
  best-taper80-surp300)   EXTRA="$AHEAD --memory-mean-factor 4 --memory-taper-ft 80 --surprise-misfit 3 300" ;;
  strong-taper80-surp300) EXTRA="$STRONG --memory-mean-factor 4 --memory-taper-ft 80 --surprise-misfit 3 300" ;;
  *) echo "unknown experiment (try: list)" >&2; exit 2 ;;
esac
NAME=fault2-$1-g$GATE-s$START
$PY -B -u sequential_matern_1d.py --reference runs/benchmark3-reference-bedding \
  --output-dir runs/$NAME --start $START --stop $STOP --ensemble-size 250 --seed $SEED --mda-steps 10 \
  --tempering linear-inflation --anisotropy log_clipped --anisotropy-mean 0.5 --anisotropy-std 0.5 \
  --memory-ft $FT $EXTRA --gate-threshold $GATE --on-gate stop --analysis enif \
  --two-d-prior ensemble_precision --directional-sigma-floor 0.5 0.25
$PY -B -u run_chain_laplace_loop.py --chain runs/$NAME --stop $STOP \
  --source references/map-source --label $NAME --map-starts 1 \
  --optimizer-setting stagnation_tolerance=0.01 --optimizer-setting stagnation_iterations=1 \
  --optimizer-setting max_rounds=2 \
  --binary-dir /cluster/home/mlie/cluster-build --orientation bedding \
  --host HPC --hpc-root /cluster/work/users/mlie/esmda-runs \
  --hpc-python /cluster/home/mlie/esmda-runs/.venv/bin/python
