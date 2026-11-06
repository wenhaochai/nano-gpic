#!/bin/bash
#SBATCH -J gpic-full-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# 8-GPU smoke of the full ladder's code (branch ladder-full, docs/ladder_plan.md section 12) on GPIC Lite shards (the full
# text cache is not written yet), with the real jobs/train_ladder.sh:
# 1. d1536 (16+4) and d1792 (18+4), the two sizes the Lite ladder never ran: 320 steps each with the step timer (step time,
#    peak memory; last.ckpt every 100 steps, so the save time shows in the step times);
# 2. stop and resume: d640 on 64 shards with 2 workers per rank (8 shards per rank and pass, about 3100 steps per pass):
#    launch 1 stops at step 1300 (last.ckpt every 200 steps), launch 2 resumes from its last.ckpt and runs to step 4800,
#    past the end of pass 0;
# 3. tools/check_shard_resume.py: the checks of the shard log and the resume, the step-time table of part 1.
# Exit 1 when any part fails. W&B runs get smoke ids, never the production ids ladder-full-<size>.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${SMOKE_WT:-$WC/project/gpic-wt/ladder_full}
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
TS=$(date +%m%d_%H%M)
SMK=$WC/runs/gpic-speed/full_smoke_$TS
mkdir -p $SMK
git -C $WT rev-parse HEAD > $SMK/commit.txt
export LADDER_WT=$WT
fail=0
log() { echo "[$(date +%T)] $*" | tee -a $SMK/phases.log; }
run() {   # run <tag> <run root> <train_ladder.sh arguments...>
  local tag=$1 root=$2; shift 2
  log "START $tag"
  LADDER_RUN_ROOT=$root bash $WT/jobs/train_ladder.sh "$@" --trainer.logger.init_args.name full_smoke_$tag \
    --trainer.logger.init_args.id full-smoke-$tag-$TS > $SMK/$tag.log 2>&1
  local rc=$?
  log "END $tag rc=$rc"
  [ $rc -eq 0 ] || fail=1
}
# 1. the two new sizes
BENCH_OUT=$SMK/bench/d1536 LADDER_PARTIAL_OK=1 LADDER_LAST_EVERY=100 run d1536 $SMK d1536 1536 16 4 3200 --trainer.max_steps 320
BENCH_OUT=$SMK/bench/d1792 LADDER_PARTIAL_OK=1 LADDER_LAST_EVERY=100 run d1792 $SMK d1792 1792 18 4 3200 --trainer.max_steps 320
# 2. stop and resume
R=$SMK/resume
DATA=(--data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..00063}.tar' --data.train_num_workers 2)
LADDER_PARTIAL_OK=1 LADDER_LAST_EVERY=200 run resume_a $R d640 640 9 2 6400 "${DATA[@]}" --trainer.max_steps 1300
mkdir -p $SMK/resume_snap && cp $R/exp_ladder_d640/shards_done_rank*.txt $R/exp_ladder_d640/launches.txt $SMK/resume_snap/ || fail=1
LADDER_PARTIAL_OK=1 LADDER_LAST_EVERY=200 run resume_b $R d640 640 9 2 6400 "${DATA[@]}" --trainer.max_steps 4800
# 3. checks
cd $WT/baselines/PixelGen
PYTHONPATH=$PWD python $WT/tools/check_shard_resume.py $SMK 2>&1 | tee -a $SMK/phases.log
[ ${PIPESTATUS[0]} -eq 0 ] || fail=1
log "DONE fail=$fail $SMK"
exit $fail
