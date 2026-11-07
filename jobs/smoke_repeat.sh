#!/bin/bash
#SBATCH -J gpic-repeat-smoke
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:30:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# 4-GPU smoke of the repetition arms (owner 2026-11-07; jobs/train_repeat.sh, EpochShuffledShards with shard copies in
# src/data/dataset/gpic_dataset.py), a whole wq block, root runs/gpic-speed/repeat_smoke/<time>. W&B off.
# 1. full: d640 on the Full path (shards 00000-07999, k = 1) for 300 steps with the step timer, the reference step time;
# 2. resume_a: d640 through jobs/train_repeat.sh on the 8-shard subset (4 ranks x 32 workers = 128 slots: 16 copies of each
#    shard per pass, one entry per worker) with the step timer: last.ckpt at step 300, stop at 400;
# 3. resume_b: the same run resumed from last.ckpt (step 300) to step 600;
# 4. tools/check_repeat_smoke.py: every worker of every rank started an entry; the median step time of the arm is within 10%
#    of the Full path's; the started-entry logs; the resumed launch skipped exactly the entries started before the checkpoint
#    and started every other entry of its pass once.
# Exit 1 when any part fails. usage: smoke_repeat.sh <worktree name under gpic-wt/, at a commit of this branch> (wq snapshots
# this script, not the code it runs)
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/${1:?usage: smoke_repeat.sh <worktree name>}
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SMK=$WC/runs/gpic-speed/repeat_smoke/$(date +%m%d_%H%M)
mkdir -p $SMK
git -C $WT rev-parse HEAD > $SMK/commit.txt
export WANDB_MODE=disabled LADDER_WT=$WT
fail=0
log() { echo "[$(date +%T)] $*" | tee -a $SMK/phases.log; }
run() {   # run <tag> <command...>
  local tag=$1; shift
  log "START $tag"
  "$@" > $SMK/$tag.log 2>&1
  local rc=$?
  log "END $tag rc=$rc"
  [ $rc -eq 0 ] || fail=1
}
DEV=(--trainer.devices 4)
# 1. the Full path's step time on the same 4 GPUs
BENCH_OUT=$SMK/bench/full LADDER_PARTIAL_OK=1 LADDER_LAST_EVERY=3200 LADDER_RUN_ROOT=$SMK/full \
  run full bash $WT/jobs/train_ladder.sh d640 640 9 2 3200 "${DEV[@]}" --trainer.max_steps 300 \
  --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
  --trainer.logger.init_args.name repeat_smoke_full --trainer.logger.init_args.id repeat-smoke-full
# 2-3. the 8-shard arm, stopped and resumed (C0 = 800 steps: ms checkpoints every 200; last.ckpt every 300)
ARM=(env REPEAT_WT=$WT REPEAT_ROOT=$SMK/arm REPEAT_LAST_EVERY=300 LADDER_PARTIAL_OK=1 bash $WT/jobs/train_repeat.sh d640 640 9 2 800 8 "${DEV[@]}")
BENCH_OUT=$SMK/bench/arm run resume_a "${ARM[@]}" --trainer.max_steps 400 \
  --trainer.logger.init_args.name repeat_smoke_a --trainer.logger.init_args.id repeat-smoke-a
mkdir -p $SMK/resume_snap && cp $SMK/arm/exp_ladder_d640/shards_started_rank*.txt $SMK/arm/exp_ladder_d640/launches.txt $SMK/resume_snap/ || fail=1
run resume_b "${ARM[@]}" --trainer.max_steps 600 \
  --trainer.logger.init_args.name repeat_smoke_b --trainer.logger.init_args.id repeat-smoke-b
# 4. checks
cd $WT/baselines/PixelGen
PYTHONPATH=$PWD python $WT/tools/check_repeat_smoke.py $SMK 2>&1 | tee -a $SMK/phases.log
[ ${PIPESTATUS[0]} -eq 0 ] || fail=1
log "DONE fail=$fail $SMK"
exit $fail
