#!/bin/bash
#SBATCH -J gpic-ladder-prof
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:20:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Where a ladder step goes on 8 H100: step time is about 41 ms + 0.14 ms per GFLOP/img across the five sizes (w920), so
# the part that no denoiser FLOP pays for is 58% of the sweep's training hours. Every arm runs the real training path
# (jobs/train_ladder.sh) for a few hundred steps with no checkpoints and no W&B.
# 1. baseline at d640, d1024, d1280: GPU span and host time of each stage (BENCH_STAGES, no profiler);
# 2. torch.profiler over 3 steps at d640 and d1280, all ranks;
# 3. d640 with one change each, same math: the text encoder without its key-value cache (28 x 2 copies and a graph break
#    that nothing reads), without the padding mask (a right-padded caption under a causal mask never sees its padding,
#    so SDPA can use the flash kernel with native GQA), both; Inductor's C++ kernel launcher; CUDA graphs in the text
#    encoder, in the text encoder and the denoiser.
# An arm that fails stays in the log and does not stop the next arm.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_prof
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/ladder_prof_$TS
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
arm() {   # arm <tag> <size> <d> <nb> <nt> <steps> [VAR=value...]
  local tag=$1 size=$2 d=$3 nb=$4 nt=$5 steps=$6; shift 6
  log "START $tag $*"
  env "$@" BENCH_OUT=$OUT/bench/$tag BENCH_STAGES=1 LADDER_RUN_ROOT=$OUT/$tag timeout 900 \
    bash $WT/jobs/train_ladder.sh $size $d $nb $nt 32000 --trainer.max_steps $steps > $OUT/$tag.log 2>&1
  log "END $tag rc=$?"
}
arm base_d640 d640 640 9 2 600
arm base_d1024 d1024 1024 12 3 400
arm base_d1280 d1280 1280 14 3 400
arm trace_d640 d640 640 9 2 160 BENCH_PROFILE=150:153
arm trace_d1280 d1280 1280 14 3 160 BENCH_PROFILE=150:153
arm te_nocache_d640 d640 640 9 2 600 GPIC_PROF_TE_NO_CACHE=1
arm te_nomask_d640 d640 640 9 2 600 GPIC_PROF_TE_NO_MASK=1
arm te_lean_d640 d640 640 9 2 600 GPIC_PROF_TE_NO_CACHE=1 GPIC_PROF_TE_NO_MASK=1
arm cpp_d640 d640 640 9 2 600 TORCHINDUCTOR_CPP_WRAPPER=1
arm te_graphs_d640 d640 640 9 2 800 GPIC_PROF_TE_NO_CACHE=1 GPIC_PROF_TE_NO_MASK=1 GPIC_PROF_TE_COMPILE_MODE=reduce-overhead
arm all_graphs_d640 d640 640 9 2 800 GPIC_PROF_TE_NO_CACHE=1 GPIC_PROF_TE_NO_MASK=1 GPIC_PROF_TE_COMPILE_MODE=reduce-overhead GPIC_PROF_DEN_COMPILE_MODE=reduce-overhead
python $WT/tools/prof/ladder_stages.py $OUT 2>&1 | tee -a $OUT/phases.log
log "DONE $OUT"
