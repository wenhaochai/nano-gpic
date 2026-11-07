#!/bin/bash
#SBATCH -J gpic-bench-cand4-r2
#SBATCH --gres=gpu:4
#SBATCH --time=01:10:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# Round 2 of exact-math candidates, 4-GPU DDP at per-GPU batch 32 (same as round 1 / w841):
#  1. benches: C3 (fused AdamW + one-pass EMA, clipping fixed), C4 (+ real RoPE), C6 (+ Flex text encoder; C5 reverted)
#  2. C6 with NCCL_PROTO=Simple, logging NCCL's algorithm/protocol choice (the all-reduces ran as RING_LL)
#  3. correctness of C6 (two records: determinism), on all 4 GPUs at once
#  4. torch.profiler window of the C6 stack for the next diagnosis
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
G=/scratch/gpfs/GROUP/USER/project/gpic-wt
OPT=$G/perf-opt
R=$OPT/workspace-runs/cand4r2_$(date +%m%d_%H%M)
mkdir -p $R
export BENCH_DEVICES=4 TORCHINDUCTOR_CACHE_DIR=$R/cache/inductor TRITON_CACHE_DIR=$R/cache/triton
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); [ ${#GPUS[@]} -ge 4 ] || GPUS=(0 1 2 3)
log() { echo "[$(date +%T)] $*" | tee -a $R/phases.log; }
phase() { local name=$1; shift; local t0=$(date +%s); log "START $name"; "$@"; local rc=$?; log "END $name rc=$rc $(( $(date +%s) - t0 ))s"; }
for c in C3 C4 C6; do phase bench_$c bash $G/cand/$c/jobs/bench_train.sh $R/bench_$c; done
phase bench_C6_nccl_simple env NCCL_PROTO=Simple NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING bash $G/cand/C6/jobs/bench_train.sh $R/bench_C6_nccl_simple
phase bench_C6_nccl_default_debug env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING BENCH_STEPS=60 bash $G/cand/C6/jobs/bench_train.sh $R/bench_C6_nccl_debug
rec() {  # rec <cand> <tag> <gpu index>
  (cd $G/cand/$1/baselines/PixelGen && env CUDA_VISIBLE_DEVICES=${GPUS[$3]} PYTHONPATH=$PWD CORR_CONFIG=$PWD/configs_h100/lite_bench.yaml \
     python $OPT/tools/correctness.py record $R/corr_$2.pt > $R/corr_$2.log 2>&1)
}
phase corr_C6 bash -c "$(declare -f rec); G=$G OPT=$OPT R=$R GPUS=(${GPUS[*]}); rec C6 C6a 0 & rec C6 C6b 1 & rec base base_r2 2 & rec C4 C4_r2 3 & wait"
phase prof_C6 bash $G/cand/C6/jobs/bench_train.sh $R/prof_C6 100:103
log "DONE $R"
