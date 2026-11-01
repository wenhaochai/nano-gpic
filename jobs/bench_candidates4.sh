#!/bin/bash
#SBATCH -J gpic-bench-cand4
#SBATCH --gres=gpu:4
#SBATCH --time=01:20:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# Candidate speedups (exact math) vs base, 4-GPU DDP at per-GPU batch 32, each from its own worktree project/gpic-wt/cand/<name>:
#  1. unprofiled bench (150 steps) per candidate, in order base, C1..C5 (cumulative commits)
#  2. correctness record per candidate (fixed batch, 10 steps, sampler), all GPUs busy in two waves
#  3. torch.profiler window (steps 100-103) of the last candidate for the next diagnosis
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
G=/scratch/gpfs/GROUP/USER/project/gpic-wt
OPT=$G/perf-opt
R=$OPT/workspace-runs/cand4_$(date +%m%d_%H%M)
mkdir -p $R
export BENCH_DEVICES=4 TORCHINDUCTOR_CACHE_DIR=$R/cache/inductor TRITON_CACHE_DIR=$R/cache/triton
GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); [ ${#GPUS[@]} -ge 4 ] || GPUS=(0 1 2 3)
log() { echo "[$(date +%T)] $*" | tee -a $R/phases.log; }
phase() { local name=$1; shift; local t0=$(date +%s); log "START $name"; "$@"; local rc=$?; log "END $name rc=$rc $(( $(date +%s) - t0 ))s"; }
CANDS="base C1 C2 C3 C4 C5"
for c in $CANDS; do phase bench_$c bash $G/cand/$c/jobs/bench_train.sh $R/bench_$c; done
rec() {  # rec <cand> <gpu index>
  (cd $G/cand/$1/baselines/PixelGen && env CUDA_VISIBLE_DEVICES=${GPUS[$2]} PYTHONPATH=$PWD CORR_CONFIG=$PWD/configs_h100/lite_bench.yaml \
     python $OPT/tools/correctness.py record $R/corr_$1.pt > $R/corr_$1.log 2>&1)
}
phase corr_wave1 bash -c "$(declare -f rec); G=$G OPT=$OPT R=$R GPUS=(${GPUS[*]}); rec base 0 & rec C1 1 & rec C2 2 & rec C3 3 & wait"
phase corr_wave2 bash -c "$(declare -f rec); G=$G OPT=$OPT R=$R GPUS=(${GPUS[*]}); rec C4 0 & rec C5 1 & wait"
phase prof_C5 bash $G/cand/C5/jobs/bench_train.sh $R/prof_C5 100:103
log "DONE $R"
