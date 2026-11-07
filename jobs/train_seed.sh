#!/bin/bash
#SBATCH -J gpic-seed
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# A second seed of the 8-step result (owner's popup 2026-11-05, docs/gmflow.md section 8): the run repeated with seed_everything <seed> and shard seed
# 1730000000 + <seed> (the first runs used 0 and 1730000000), the official crop, everything else the same, then FD at C0/16 and C0/8 at 8 and 50 calls.
# K = 0: the point model w1147 with its own frozen code (gpic-wt/ar_real, 66816 steps, no caption dropout), evaluated with the official Euler.
# K = 8: GMFlow w1173 with gpic-wt/gmflow_b2 (gmflow_run2's code; 62336 steps), evaluated with GM-ODE.
# Resumable (wq --partial): the training resumes from last.ckpt, finished evaluations are skipped.
# usage: train_seed.sh <K: 0 | 8> <seed: 1 to 9>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
[ $# -eq 2 ] || { echo "usage: train_seed.sh <0 | 8> <seed>"; exit 2; }
K=$1 SEED=$2
case $SEED in [1-9]) ;; *) echo "seed must be 1 to 9"; exit 2 ;; esac
case $K in
  0) WT=$WC/project/gpic-wt/ar_real; STEPS=66816; NAME=point_nodrop_d640_s$SEED; EXTRA=(); S=euler
     grep -q "null_condition_p: 0.0" $WT/baselines/PixelGen/configs_h100/lite_ladder.yaml || { echo "$WT still has caption dropout"; exit 2; } ;;
  8) WT=$WC/project/gpic-wt/gmflow_b2; STEPS=62336; NAME=gmflow_k8_d640_s$SEED; EXTRA=(--config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k 8); S=gm_ode ;;
  *) echo "K must be 0 or 8"; exit 2 ;;
esac
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
ROOT=$WC/runs/gpic-speed/$NAME
mkdir -p $ROOT
SHA=$(git -C $WT rev-parse HEAD)
if [ -f $ROOT/commit.txt ] && [ "$(cat $ROOT/commit.txt)" != "$SHA" ]; then echo "$ROOT was started at $(cat $ROOT/commit.txt), the worktree holds $SHA"; exit 2; fi
echo $SHA > $ROOT/commit.txt
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT GPIC_SHARD_SEED=$((1730000000 + SEED)) GPIC_REGION_CROP=0
if ls $ROOT/exp_ladder_d640/ms-step=$(printf %06d $STEPS)*.ckpt > /dev/null 2>&1; then
  echo "training done before"
else
  bash $WT/jobs/train_ladder.sh d640 640 9 2 $STEPS "${EXTRA[@]}" --seed_everything $SEED \
    --trainer.logger.init_args.name $NAME --trainer.logger.init_args.id ${NAME//_/-} >> $ROOT/train.log 2>&1
  rc=$?
  echo "train rc=$rc launches=$(wc -l < $ROOT/exp_ladder_d640/launches.txt 2>/dev/null)"
  [ $rc -eq 0 ] || exit $rc
fi
H=$((STEPS / 2))
bash $WC/project/gpic-wt/gmflow_run2/jobs/evals_gmflow.sh $K d640 640 9 2 $ROOT $H:$S:8 $STEPS:$S:8 $H:$S:50 $STEPS:$S:50
