#!/bin/bash
#SBATCH -J gpic-bench-cand4-r3
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# Round 3, 4-GPU DDP: C7 (no host sync in the step) at per-GPU batch 32 and 64 (64 x 4 = the official global batch
# 256), C7 correctness (two records, determinism) against a base record of this job, and a profile window of C7.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
G=/scratch/gpfs/GROUP/USER/project/gpic-wt
OPT=$G/perf-opt
R=$OPT/workspace-runs/cand4r3_$(date +%m%d_%H%M)
mkdir -p $R
export BENCH_DEVICES=4 TORCHINDUCTOR_CACHE_DIR=$R/cache/inductor TRITON_CACHE_DIR=$R/cache/triton
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); [ ${#GPUS[@]} -ge 4 ] || GPUS=(0 1 2 3)
log() { echo "[$(date +%T)] $*" | tee -a $R/phases.log; }
phase() { local name=$1; shift; local t0=$(date +%s); log "START $name"; "$@"; local rc=$?; log "END $name rc=$rc $(( $(date +%s) - t0 ))s"; }
phase bench_C7 bash $G/cand/C7/jobs/bench_train.sh $R/bench_C7
phase bench_C7_b64 env BENCH_STEPS=100 BENCH_ARGS="--data.train_batch_size 64" bash $G/cand/C7/jobs/bench_train.sh $R/bench_C7_b64
rec() {  # rec <cand> <tag> <gpu index>
  (cd $G/cand/$1/baselines/PixelGen && env CUDA_VISIBLE_DEVICES=${GPUS[$3]} PYTHONPATH=$PWD CORR_CONFIG=$PWD/configs_h100/lite_bench.yaml \
     python $OPT/tools/correctness.py record $R/corr_$2.pt > $R/corr_$2.log 2>&1)
}
phase corr_C7 bash -c "$(declare -f rec); G=$G OPT=$OPT R=$R GPUS=(${GPUS[*]}); rec C7 C7a 0 & rec C7 C7b 1 & rec base base_r3 2 & wait"
phase prof_C7 bash $G/cand/C7/jobs/bench_train.sh $R/prof_C7 100:103
log "DONE $R"
