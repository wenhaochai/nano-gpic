#!/bin/bash
#SBATCH -J gpic-ar
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The autoregressive model at d640 to C0/8, then FD at C0/16 and C0/8 (docs/autoregressive.md). Same compute as the point model: 133.00 GFLOP per image against 104.15 (tools/ar/test_ar.py), so 66816 x 104.15 / 133.00 = 52322, and 52320 steps (a multiple of 32).
# Rule, set before the run: valid only if FD is below the point model with each prompt's own noise, 1529.61 at C0/16 and 1083.94 at C0/8 (w1098, w1099), by more than 5% at both points. The seed test shows 2-4% between two runs of one recipe.
# Code: the frozen worktree gpic-wt/ar_run. Same shard order as the baseline, w1053.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ar_run
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
ROOT=$WC/runs/gpic-speed/ar_d640
mkdir -p $ROOT
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT
export GPIC_SHARD_SEED=1730000000
bash $WT/jobs/train_ladder.sh d640 640 9 2 52320 --config configs_h100/ar_train.yaml --trainer.logger.init_args.name ar_d640 --trainer.logger.init_args.id ar-d640 >> $ROOT/train.log 2>&1
rc=$?
echo "train rc=$rc"
[ $rc -eq 0 ] || exit $rc
fail=0
for STEP in 26160 52320; do
  bash $WT/jobs/eval_ar.sh d640 640 9 2 $STEP > $ROOT/eval_$STEP.log 2>&1 || fail=1
  echo "eval $STEP fail=$fail"
done
exit $fail
