#!/bin/bash
#SBATCH -J gpic-ladder-pin2
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:35:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Why did the pin arm of w1048 stall? Main and autograd threads on one shared CPU: median 25.0 ms, but 21 steps over 0.2 s, each one rank waiting for its batch, and the workers' wait for a CPU
# 37 CPU-equivalents at the same CPU use as without pinning, so workers waited while CPUs were idle. Hypothesis: a run queue with two pinned threads is seen as the busiest, a balance pass that
# cannot pull them backs off (LBF_ALL_PINNED). Test: main and autograd on one CPU each (GPIC_PROF_PIN=2), per-CPU idle and wait (tools/prof/cpu_wait.py), and the loader's buffers per step (GPIC_PROF_QUEUE=1).
# Adoption rule, set before the run: adopt only if the mean at d640 is below both d640 controls by at least 3%, with no more steps over 0.3 s than the controls, and the mean at d1280 is at most 0.5% above its control.
# On 8 H100s, on one node. Measures only. No W&B. Same math. Expected: about 20 min.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_pin2
[ -d "$WT" ] || { echo "no worktree $WT"; exit 2; }
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/ladder_pin2_$TS
PROF=$WC/tmp/ladder_prof/host_prof/ladder_pin2_$TS
mkdir -p $OUT $PROF
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled GPIC_PROF_QUEUE=1
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }

arm() {   # arm <tag> <limit s> <size> <d> <nb> <nt> <steps of this launch> [main.py arguments...]
  local tag=$1 limit=$2 size=$3 d=$4 nb=$5 nt=$6 stop=$7; shift 7
  log "START $tag"
  BENCH_OUT=$OUT/bench/$tag LADDER_RUN_ROOT=$OUT/$tag timeout -k 60 $limit \
    bash $WT/jobs/train_ladder.sh $size $d $nb $nt 32000 --trainer.max_steps $stop "$@" > $OUT/$tag.log 2>&1
  log "END $tag rc=$?"
}
d640() { local tag=$1 limit=$2 stop=$3; shift 3; arm $tag $limit d640 640 9 2 $stop "$@"; }
d1280() { local tag=$1 limit=$2 stop=$3; shift 3; arm $tag $limit d1280 1280 14 3 $stop "$@"; }
wait_arm() {   # wait_arm <tag>: 20 s after the warm-up, the threads of rank 3 over 15 s and the CPUs and each rank's workers over 30 s
  mkdir -p $PROF/$1
  until grep -q "\[warmup\] done" $OUT/$1.log 2>/dev/null; do
    grep -q "END $1 " $OUT/phases.log && return
    sleep 2
  done
  sleep 20
  python3 -I $WT/tools/prof/thread_wait.py 3 15 > $PROF/$1/thread_wait.txt 2>&1 &
  python3 -I $WT/tools/prof/cpu_wait.py 30 > $PROF/$1/cpu_wait.txt 2>&1 &
  wait
}

wait_arm d640_control &
d640 d640_control 300 2400
wait
wait_arm d640_pin &
GPIC_PROF_PIN=1 d640 d640_pin 300 2400
wait
wait_arm d640_pin_split &
GPIC_PROF_PIN=2 d640 d640_pin_split 300 2400
wait
wait_arm d640_control2 &
d640 d640_control2 300 2400
wait
d1280 d1280_control 420 800
GPIC_PROF_PIN=2 d1280 d1280_pin_split 420 800

log "DONE $OUT"
