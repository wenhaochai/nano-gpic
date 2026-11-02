#!/bin/bash
#SBATCH -J gpic-c7-bisect8
#SBATCH --gres=gpu:8
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# C7 ran at 707 ms/step (C6: 213 ms). Bisect its three changes: full C7 and C7 without one change each, as 4-GPU
# DDP benches two at a time (GPUs 0-3 and 4-7 of this job; co-location left C6 unchanged in w845).
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
G=/scratch/gpfs/GROUP/USER/project/gpic-wt
R=$G/perf-opt/workspace-runs/c7bisect_$(date +%m%d_%H%M)
mkdir -p $R
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); [ ${#GPUS[@]} -ge 8 ] || GPUS=(0 1 2 3 4 5 6 7)
A="${GPUS[0]},${GPUS[1]},${GPUS[2]},${GPUS[3]}"; B="${GPUS[4]},${GPUS[5]},${GPUS[6]},${GPUS[7]}"
log() { echo "[$(date +%T)] $*" | tee -a $R/phases.log; }
run() {  # run <cand dir> <tag> <gpus> <port>
  log "START $2"
  env CUDA_VISIBLE_DEVICES=$3 MASTER_PORT=$4 BENCH_DEVICES=4 BENCH_STEPS=80 TORCHINDUCTOR_CACHE_DIR=$R/cache/$2/inductor \
      TRITON_CACHE_DIR=$R/cache/$2/triton bash $G/cand/$1/jobs/bench_train.sh $R/$2
  log "END $2 rc=$?"
}
run C7 C7 $A 29511 & run C7_nosig C7_nosig $B 29512 & wait
run C7_nopin C7_nopin $A 29513 & run C7_norope C7_norope $B 29514 & wait
log "DONE $R"
