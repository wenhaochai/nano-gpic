#!/bin/bash
#SBATCH -J gpic-ladder-pin3
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:42:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# w1093 found why the pin arm stalls: at a stall q_ready is 0 and q_raw is 64-88, so the workers have finished the batches and the rank's pin thread has not copied them. Its wait for a CPU is 42-51% of the wall clock
# (15% without pinning), and the main thread's is 0.4%. Under SCHED_IDLE workers, the pin thread's wait was 3.6% (w1055). So: pinning and SCHED_IDLE workers together, on the code of w1093.
# Adoption rule, set before the run. Measured on steps 100-2400 without save steps (tmp/ladder_prof/arm_table.py). Adopt only if the mean at d640 is below both d640 controls by at least 3%, no more steps over 0.3 s than the larger count of the two controls,
# and the mean at d1280 at most 0.5% above its control (the noise level, so a result near it is weak evidence).
# On 8 H100s, on one node. Measures only. No W&B. Same math. Expected: about 20 min.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_pin2
[ -d "$WT" ] || { echo "no worktree $WT"; exit 2; }
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/ladder_pin3_$TS
PROF=$WC/tmp/ladder_prof/host_prof/ladder_pin3_$TS
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
wait_arm() {   # wait_arm <tag>: 40 s after the warm-up (past the save at step 1000), the threads of rank 3 over 15 s and the CPUs and each rank's workers over 25 s, off the reserved CPUs 80-95
  mkdir -p $PROF/$1
  until grep -q "\[warmup\] done" $OUT/$1.log 2>/dev/null; do
    grep -q "END $1 " $OUT/phases.log && return
    sleep 2
  done
  sleep 40
  taskset -c 0-79 python3 -I $WT/tools/prof/thread_wait.py 3 15 > $PROF/$1/thread_wait.txt 2>&1 &
  taskset -c 0-79 python3 -I $WT/tools/prof/cpu_wait.py 25 > $PROF/$1/cpu_wait.txt 2>&1 &
  wait
}

wait_arm d640_control &
d640 d640_control 300 2400
wait
wait_arm d640_pin_idle &
GPIC_PROF_PIN=1 GPIC_WORKER_IDLE=1 d640 d640_pin_idle 300 2400
wait
wait_arm d640_split_idle &
GPIC_PROF_PIN=2 GPIC_WORKER_IDLE=1 d640 d640_split_idle 300 2400
wait
wait_arm d640_control2 &
d640 d640_control2 300 2400
wait
d1280 d1280_control 420 800
GPIC_PROF_PIN=1 GPIC_WORKER_IDLE=1 d1280 d1280_pin_idle 420 800

log "DONE $OUT"
