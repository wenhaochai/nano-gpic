#!/bin/bash
#SBATCH -J gpic-ladder-seed
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Does a different seed per rank help? The official recipe seeds all ranks alike, so a global batch of 256 has 32 different (t, noise). d640 to C0/8
# (66816 steps, a multiple of 32), then the ladder's evaluation at C0/16 and C0/8. Owner's popup 2026-11-03: test before any change to the ladder.
# usage: smoke_ladder_seed.sh same|rank     Code: the frozen worktree gpic-wt/ladder_seed.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
ARM=$1
case "$ARM" in same) ;; rank) export GPIC_RANK_SEED=1 ;; *) echo "usage: same|rank"; exit 2 ;; esac
WT=$WC/project/gpic-wt/ladder_seed
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
ROOT=$WC/runs/gpic-speed/ladder_seed_$ARM
mkdir -p $ROOT
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT
bash $WT/jobs/train_ladder.sh d640 640 9 2 66816 --trainer.logger.init_args.name seed_$ARM --trainer.logger.init_args.id seed-$ARM > $ROOT/train.log 2>&1
rc=$?
echo "train rc=$rc"
[ $rc -eq 0 ] || exit $rc
for STEP in 33408 66816; do
  bash $WT/jobs/eval_ladder.sh d640 640 9 2 $STEP > $ROOT/eval_$STEP.log 2>&1
  echo "eval $STEP rc=$?"
done
