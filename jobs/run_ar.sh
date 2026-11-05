#!/bin/bash
#SBATCH -J gpic-ar
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Test C: the autoregressive model in MAR's order at d640 to C0/8, then FD at C0/16 and C0/8, as drawn and with the first 64 tokens drawn again (docs/autoregressive.md, section 测试 C). Same compute as the point model: 133.00 GFLOP per image against 104.15 (tools/ar/test_ar.py), so C0/8 is 52320 steps and C0/16 26160. No caption dropout (owner 2026-11-04). Its baseline is jobs/run_point_nodrop.sh, not w1053, which has caption dropout.
# A known bias against this model at C0/16: at the same compute it takes fewer steps, so the EMA of 0.9999 still holds 7.3% of the initial weights there, against 3.5% for the point model.
# Code: the frozen worktree gpic-wt/ar_real. Same shard order and the same official crop as w1053.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ar_real
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q "redraw_first" $WT/baselines/PixelGen/src/models/transformer/AR_JiT.py && grep -q "order: random" $WT/baselines/PixelGen/configs_h100/ar_train.yaml || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
ROOT=$WC/runs/gpic-speed/ar_mar_d640
mkdir -p $ROOT
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT
export GPIC_SHARD_SEED=1730000000 GPIC_REGION_CROP=0
bash $WT/jobs/train_ladder.sh d640 640 9 2 52320 --config configs_h100/ar_train.yaml --trainer.logger.init_args.name ar_mar_d640 --trainer.logger.init_args.id ar-mar-d640 >> $ROOT/train.log 2>&1
rc=$?
echo "train rc=$rc"
[ $rc -eq 0 ] || exit $rc
fail=0
for STEP in 26160 52320; do
  for V in base first64; do
    bash $WT/jobs/eval_ar.sh d640 640 9 2 $STEP $V > $ROOT/eval_${STEP}_$V.log 2>&1 || fail=1
    echo "eval $STEP $V fail=$fail"
  done
done
exit $fail
