#!/bin/bash
#SBATCH -J gpic-ladder-host
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Host levers on 8 H100s, on the cached path with the C++ kernel launcher. All keep the math; each arm changes one thing against
# the control arm of its size. At d640 the step is bound by the host: on-CPU time is mostly Python and malloc, 11% in the CUDA
# driver (tmp/ladder_prof/host_prof). At d1280 it is bound by the GPU, so this arm set shows what the levers do there.
# Arms: Lightning's per-step SIGTERM broadcast off, gc.freeze after the warm-up, log_dict only on steps that write logs, tcmalloc,
# the rank and its workers on the GPU's socket, DDP buckets of 100 MB and 1000 MB (fewer all-reduces, fewer DDPOptimizer subgraphs).
# Measures only. No W&B. Historical: run at ee4b972 as w974. The switches it sets are gone since a5ab18d: HostTuning, tcmalloc and
# log_dict only on written steps are now the defaults of jobs/train_ladder.sh, and Lightning's broadcast stays (off changes nothing).
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_cache
[ -d "$WT" ] || { echo "no worktree $WT"; exit 2; }
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/ladder_host_$TS
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles
TCMALLOC=/usr/lib64/libtcmalloc.so.4
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
arm() {   # arm <tag> <size> <d> <nb> <nt> <steps of this launch> [main.py arguments...]
  local tag=$1 size=$2 d=$3 nb=$4 nt=$5 stop=$6; shift 6
  log "START $tag"
  BENCH_OUT=$OUT/bench/$tag BENCH_STAGES=${ARM_STAGES-1} LADDER_RUN_ROOT=$OUT/$tag timeout 1500 \
    bash $WT/jobs/train_ladder.sh $size $d $nb $nt 32000 --trainer.max_steps $stop "$@" > $OUT/$tag.log 2>&1
  log "END $tag rc=$?"
}
[ -f $TCMALLOC ] || log "no $TCMALLOC on $(hostname): the tcmalloc arms measure nothing"
B=--trainer.strategy.init_args.bucket_cap_mb
light() { GPIC_PROF_NO_SIGTERM_BCAST=1 GPIC_PROF_GC_FREEZE=1 GPIC_PROF_LOG_ONLY_WHEN_WRITTEN=1 GPIC_PROF_NUMA=1 LD_PRELOAD=$TCMALLOC "$@"; }
d640() { arm "$1" d640 640 9 2 "${2:-1400}" "${@:3}"; }
d1280() { arm "$1" d1280 1280 14 3 "${2:-700}" "${@:3}"; }

# d640: 3400 steps where gen-2 collections matter (they start near step 2900), 1400 otherwise
d640 d640_control 3400
GPIC_PROF_GC_FREEZE=1 d640 d640_gc 3400
GPIC_PROF_NO_SIGTERM_BCAST=1 d640 d640_nobcast
GPIC_PROF_LOG_ONLY_WHEN_WRITTEN=1 d640 d640_log
LD_PRELOAD=$TCMALLOC d640 d640_tcmalloc
GPIC_PROF_NUMA=1 d640 d640_numa
d640 d640_b100 1400 $B 100
d640 d640_b1000 1400 $B 1000
light d640 d640_light
light d640 d640_light_b100 1400 $B 100
ARM_STAGES= light d640 d640_light_nostages

# d1280
d1280 d1280_control
d1280 d1280_b100 700 $B 100
d1280 d1280_b1000 700 $B 1000
light d1280 d1280_light

python $WT/tools/prof/ladder_stages.py $OUT 2>&1 | tee -a $OUT/phases.log
log "DONE $OUT"
