#!/bin/bash
#
# Slurm orchestration for the 2D ESMDA gradient phase.
#
# Submits three chained jobs:
#   1. prep    (serial 0D+1D ESMDA, writes stage_2d.npz)            -- light, 1 task
#   2. array   (one task per 2D ensemble member, internal MPI)       -- heavy, --array=0..NE2D-1
#   3. collect (gather member_*.pkl, plot, save results pkl)         -- light, 1 task
#
# Jobs 2 and 3 depend (afterok) on the previous stage, so the whole chain runs
# from a single `bash run_da_slurm.sh` invocation with no manual job-id handoff.
#
# Usage:
#   bash run_da_slurm.sh
#
# Tunables (edit below or override via env): REF, DATA_DIR, NE2D,
# MPI_RANKS_PER_MEMBER, OMP, PREP_TIME, ARRAY_TIME.
#
# NOTE on internal MPI: each array task sets EMSIM_MPI_PREFIX so EMsim launches
# the Fortran binary via `srun --mpi=pmi2`, using the task's allocated ranks.
# EMsim's call_sim format is private (NORCE); if `options['mpi']` is not honored
# the binary runs single-process and you should fall back to OpenMP-only
# (set MPI_RANKS_PER_MEMBER=1 and raise OMP). Confirm on the HPC first.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override with env vars)
# ---------------------------------------------------------------------------
REF="${REF:-/home/AD.NORCERESEARCH.NO/mlie/3DGiG/Jacobian/inversion/data/Benchmark-3}"
DATA_DIR="${DATA_DIR:-$PWD}"                 # where data.pkl / var.pkl live
RESULTS="${RESULTS:-$PWD/results_da}"
WORK_ROOT="${WORK_ROOT:-$PWD/work_da}"
STAGE="$RESULTS/stage_2d.npz"

NE2D="${NE2D:-10}"                           # number of 2D ensemble members
NE01="${NE01:-250}"                          # 0D/1D ensemble size
MPI_RANKS_PER_MEMBER="${MPI_RANKS_PER_MEMBER:-20}"   # K: Fortran MPI ranks per 2D member
OMP="${OMP:-5}"                              # OpenMP threads per rank
SEED="${SEED:-42}"
ASSIM_INDEX="${ASSIM_INDEX:-180}"

PREP_TIME="${PREP_TIME:-06:00:00}"
ARRAY_TIME="${ARRAY_TIME:-48:00:00}"
COLLECT_TIME="${COLLECT_TIME:-00:30:00}"

PREP_MEM="${PREP_MEM:-64G}"
MEM_PER_MEMBER="${MEM_PER_MEMBER:-120G}"
COLLECT_MEM="${COLLECT_MEM:-16G}"

PY="${PY:-python}"                           # python interpreter with EMsim/pipt/etc

# ---------------------------------------------------------------------------
# Shared environment hook (Intel oneAPI + HDFql + MPI/PMI2 + stack)
# Sourced inside each job via --wrap. Mirrors run_bench.sh.
# ---------------------------------------------------------------------------
read -r -d '' ENV_HOOK <<'ENVEOF' || true
source ~/intel/oneapi/2025.2/oneapi-vars.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/LogSimulator3D_v4.2fullfortran_jacobi_global/src/hdfql_linux/"
ulimit -s unlimited
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
ENVEOF

mkdir -p "$RESULTS" "$WORK_ROOT"
echo "REF=$REF"
echo "DATA_DIR=$DATA_DIR"
echo "RESULTS=$RESULTS  WORK_ROOT=$WORK_ROOT"
echo "NE2D=$NE2D  MPI_RANKS_PER_MEMBER=$MPI_RANKS_PER_MEMBER  OMP=$OMP"

# ---------------------------------------------------------------------------
# Job 1: prep (0D + 1D ESMDA -> stage_2d.npz). Single task, light.
# ---------------------------------------------------------------------------
JOB1=$(sbatch --parsable \
    --job-name=da-prep \
    --time="$PREP_TIME" --ntasks=1 --cpus-per-task=4 --mem="$PREP_MEM" \
    --output="$RESULTS/prep-%j.log" \
    --wrap="$ENV_HOOK
export OMP_NUM_THREADS=4
$PY run_DA.py --mode prep \
    --reference-model-path '$REF' \
    --data-dir '$DATA_DIR' \
    --results-dir '$RESULTS' \
    --stage-file '$STAGE' \
    --ne '$NE01' --ne-2d '$NE2D' \
    --assim-index '$ASSIM_INDEX' --seed '$SEED'")
echo "prep job: $JOB1"

# ---------------------------------------------------------------------------
# Job 2: array (one 2D member per task, internal MPI). Depends on prep.
#   --array=0..NE2D-1, each task gets MPI_RANKS_PER_MEMBER ranks x OMP threads.
# ---------------------------------------------------------------------------
ARRAY_MAX=$((NE2D - 1))
JOB2=$(sbatch --parsable \
    --job-name=da-2d-%a \
    --array="0-$ARRAY_MAX" \
    --time="$ARRAY_TIME" \
    --ntasks="$MPI_RANKS_PER_MEMBER" --cpus-per-task="$OMP" \
    --mem="$MEM_PER_MEMBER" \
    --dependency=afterok:"$JOB1" \
    --output="$RESULTS/member-%A_%a.log" \
    --wrap="$ENV_HOOK
export OMP_NUM_THREADS=$OMP
export EMSIM_MPI_PREFIX='srun --mpi=pmi2'
export RESULTS_DIR='$RESULTS'
export WORK_ROOT='$WORK_ROOT'
$PY run_DA.py --mode worker \
    --member \$SLURM_ARRAY_TASK_ID \
    --reference-model-path '$REF' \
    --results-dir '$RESULTS' \
    --work-root '$WORK_ROOT' \
    --stage-file '$STAGE' \
    --seed '$SEED'")
echo "array job: $JOB2  (afterok:$JOB1)"

# ---------------------------------------------------------------------------
# Job 3: collect (assemble results, plot, save inversion_results pkl).
#        Depends on the whole array finishing OK.
# ---------------------------------------------------------------------------
JOB3=$(sbatch --parsable \
    --job-name=da-collect \
    --time="$COLLECT_TIME" --ntasks=1 --mem="$COLLECT_MEM" \
    --dependency=afterok:"$JOB2" \
    --output="$RESULTS/collect-%j.log" \
    --wrap="$ENV_HOOK
$PY run_DA.py --mode collect \
    --results-dir '$RESULTS' \
    --stage-file '$STAGE'")
echo "collect job: $JOB3  (afterok:$JOB2)"

echo
echo "Chain submitted: prep=$JOB1 -> array=$JOB2 -> collect=$JOB3"
echo "Monitor:  squeue -u \$USER        Status: sacct -j $JOB1,$JOB2,$JOB3 --format=JobID,JobName,ArrayTaskID,State,Elapsed"
