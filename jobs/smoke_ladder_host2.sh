#!/bin/bash
#SBATCH -J gpic-ladder-host2
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The levers left after w1048 (docs/text_cache.md, "What is left of the step"), on 8 H100s, on one node. Measures only. No W&B.
# d640: data workers at a higher nice value; fewer data workers. d1280: one DDP bucket of 3000 MiB instead of three (the graphs change,
# so that arm compiles cold). Same math, except that the number of workers changes the order of the data. The SIGTERM broadcast has no
# arm: its cost per broadcast step comes from the per-step times of the controls (review finding 3). A second d640 control at the end
# gives the spread between launches. Expected: about 30 min.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_host2
[ -d "$WT" ] || { echo "no worktree $WT"; exit 2; }
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/ladder_host2_$TS
PROF=$WC/tmp/ladder_prof/host_prof/ladder_host2_$TS
mkdir -p $OUT $PROF
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled
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
wait_arm() {   # wait_arm <tag>: 20 s after the warm-up, CPU use and wait of rank 3 over 15 s, and the nice values of the threads
  mkdir -p $PROF/$1
  until grep -q "\[warmup\] done" $OUT/$1.log 2>/dev/null; do
    grep -q "END $1 " $OUT/phases.log && return
    sleep 2
  done
  sleep 20
  python3 -I $WT/tools/prof/thread_wait.py 3 15 > $PROF/$1/thread_wait.txt 2>&1
  ps -u $(id -u) -L -o ni=,comm= | sort | uniq -c | sort -rn | head -12 > $PROF/$1/nice.txt
}

wait_arm d640_control &
d640 d640_control 420 2400
wait
d1280 d1280_control 420 800
d1280 d1280_b3000 720 800 --trainer.strategy.init_args.bucket_cap_mb 3000
wait_arm d640_nice10 &
GPIC_WORKER_NICE=10 d640 d640_nice10 420 2400
wait
wait_arm d640_w24 &
d640 d640_w24 420 2400 --data.train_num_workers 24
wait
wait_arm d640_w16 &
d640 d640_w16 420 2400 --data.train_num_workers 16
wait
wait_arm d640_w16_nice10 &
GPIC_WORKER_NICE=10 d640 d640_w16_nice10 420 2400 --data.train_num_workers 16
wait
d640 d640_control2 420 2400

log "DONE $OUT"
