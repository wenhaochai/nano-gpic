#!/bin/bash
#SBATCH -J gpic-ladder-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# 8-GPU smoke of the ladder sweep with the real job scripts and the frozen worktree gpic-wt/ladder_run:
# 1. every size for 320 steps with the step timer (step time, peak memory, compiled graphs); d640 first to 160 steps,
#    then resumed from last.ckpt to 320;
# 2. epoch ends: d640 on 16 shards with 2 workers per rank (one shard per worker, ~790 steps per epoch) for 2400 steps,
#    so every rank passes 2-3 epoch ends at its own step;
# 3. the eval path on the d640 checkpoint at step 320;
# 4. tools/check_ladder_smoke.py: checkpoints, resume, shard order per epoch, eval output, the step-time table.
# Exit 1 when any part fails. W&B runs get smoke ids, never the production ids ladder-<size>.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_run
TS=$(date +%m%d_%H%M)
SMK=$WC/runs/gpic-speed/ladder_smoke_$TS
mkdir -p $SMK
git -C $WT rev-parse HEAD > $SMK/commit.txt
export LADDER_WT=$WT LADDER_RUN_ROOT=$SMK
export TORCH_LOGS=recompiles   # the checker counts the recompilations after the warm-up
fail=0
log() { echo "[$(date +%T)] $*" | tee -a $SMK/phases.log; }
run() {   # run <tag> <wandb id suffix> <train_ladder.sh arguments...>
  local tag=$1 wid=$2; shift 2
  log "START $tag"
  BENCH_OUT=$SMK/bench/$tag bash $WT/jobs/train_ladder.sh "$@" --trainer.logger.init_args.name ladder_smoke_$wid \
    --trainer.logger.init_args.id ladder-smoke-$wid-$TS > $SMK/$tag.log 2>&1
  local rc=$?
  log "END $tag rc=$rc"
  [ $rc -eq 0 ] || fail=1
}
run d640_a d640 d640 640 9 2 320 --trainer.max_steps 160
run d640_b d640 d640 640 9 2 320
run d768 d768 d768 768 10 2 320
run d896 d896 d896 896 11 2 320
run d1024 d1024 d1024 1024 12 3 320
run d1280 d1280 d1280 1280 14 3 320
export LADDER_RUN_ROOT=$SMK/epoch
run epoch epoch d640 640 9 2 2400 --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..00015}.tar' \
  --data.train_num_workers 2
export LADDER_RUN_ROOT=$SMK
log "START eval"
bash $WT/jobs/eval_ladder.sh d640 640 9 2 320 > $SMK/eval.log 2>&1; rc=$?
log "END eval rc=$rc"; [ $rc -eq 0 ] || fail=1
python $WT/tools/check_ladder_smoke.py $SMK $WT/docs/ladder_runs.json 2>&1 | tee -a $SMK/phases.log; [ ${PIPESTATUS[0]} -eq 0 ] || fail=1
log "DONE fail=$fail $SMK"
exit $fail
