#!/bin/bash
#SBATCH -J gpic-point-nodrop
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The control of test C: the point model at d640 to C0/8 (66816 steps), with no caption dropout (owner 2026-11-04), then FD at C0/16 and C0/8 with each prompt's own noise. Same shard order and the same official crop as w1053, which differs only by the dropout, so this also measures the dropout. docs/autoregressive.md, section 测试 C. Code: the frozen worktree gpic-wt/ar_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ar_run
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q "null_condition_p: 0.0" $WT/baselines/PixelGen/configs_h100/lite_ladder.yaml || { echo "$WT still has caption dropout"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
ROOT=$WC/runs/gpic-speed/point_nodrop_d640
mkdir -p $ROOT
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT
export GPIC_SHARD_SEED=1730000000 GPIC_REGION_CROP=0
bash $WT/jobs/train_ladder.sh d640 640 9 2 66816 --trainer.logger.init_args.name point_nodrop_d640 --trainer.logger.init_args.id point-nodrop-d640 >> $ROOT/train.log 2>&1
rc=$?
echo "train rc=$rc"
[ $rc -eq 0 ] || exit $rc
fail=0
for STEP in 33408 66816; do
  EVAL_NOISE=prompt bash $WT/jobs/eval_ladder.sh d640 640 9 2 $STEP > $ROOT/eval_$STEP.log 2>&1 || fail=1
  echo "eval $STEP fail=$fail"
done
exit $fail
