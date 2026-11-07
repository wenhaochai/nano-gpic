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
rec() {  # rec <cand> <tag> <gpu index> [extra env]: one correctness record of candidate <cand>
  (cd $G/cand/$1/baselines/PixelGen && env CUDA_VISIBLE_DEVICES=${GPUS[$3]} PYTHONPATH=$PWD CORR_CONFIG=$PWD/configs_h100/lite_bench.yaml $4 \
     python $OPT/tools/correctness.py record $R/corr_$2.pt > $R/corr_$2.log 2>&1)
}
# C1, C2 change data movement, logging and DDP config only: the fixed-batch recorder cannot see them, by construction.
phase corr_wave1 bash -c "$(declare -f rec); G=$G OPT=$OPT R=$R GPUS=(${GPUS[*]}); rec base base1 0 & rec base base2 1 & rec base base3 2 & rec base sdpa_math 3 CORR_SDPA=math & wait"
phase corr_wave2 bash -c "$(declare -f rec); G=$G OPT=$OPT R=$R GPUS=(${GPUS[*]}); rec C3 C3 0 & rec C4 C4 1 & rec C5 C5 2 & wait"
phase corr_derive bash -c "python $OPT/tools/correctness.py derive $R/tol3.json $R/corr_base1.pt $R/corr_base2.pt $R/corr_base3.pt > $R/tol3.log 2>&1; python $OPT/tools/correctness.py derive $R/tol_swap.json $R/corr_base1.pt $R/corr_base2.pt $R/corr_base3.pt $R/corr_sdpa_math.pt > $R/tol_swap.log 2>&1; for c in C3 C4 C5; do echo == \$c; python $OPT/tools/correctness.py compare $R/tol_swap.json $R/corr_base1.pt $R/corr_\$c.pt; done > $R/compare.log 2>&1; true"
phase prof_C5 bash $G/cand/C5/jobs/bench_train.sh $R/prof_C5 100:103
log "DONE $R"
