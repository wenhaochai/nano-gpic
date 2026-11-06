#!/bin/bash
#SBATCH -J gpic-gmflow-train
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=06:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# GMFlow training at any width and compute (docs/gmflow.md section 6, owner 2026-11-05): the Gaussian-mixture head with the transition loss (lambda 0.5) and K
# components, FLOP-matched to the point model of the same size: STEPS is the largest multiple of 128 with STEPS (1 + r) <= the point model's steps, r = the head's
# added forward FLOPs / the ladder's forward FLOPs per image (docs/ladder_runs.json gflop_per_img / 3). Resumable (wq --partial): train_ladder.sh resumes from
# last.ckpt and exits 0 only when done; a run whose last ms checkpoint exists skips the training.
# data: ladder = train_ladder.sh's defaults (shard order per launch, the region crop), as the ladder's point models; w1147 = the fixed shard seed and the official
# crop of w1147, w1173 and w1174 (the C0/8 comparison).
# Optional evaluation tasks after the training (<step>:<sampler>:<calls>, jobs/evals_gmflow.sh); long runs evaluate in separate jobs.
# usage: train_gmflow.sh <K> <size> <d> <nb> <nt> <steps> <tag> <data: ladder | w1147> [step:sampler:calls ...]
# Output: runs/gpic-speed/gmflow_k<K>_<size>_<tag>/exp_ladder_<size>; W&B gmflow-k<K>-<size>-<tag>. Code: the frozen worktree gpic-wt/gmflow_run2; the run records its
# commit and refuses a worktree that moved.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
[ $# -ge 8 ] || { echo "usage: train_gmflow.sh <K> <size> <d> <nb> <nt> <steps> <tag> <ladder | w1147> [tasks...]"; exit 2; }
K=$1 SIZE=$2 WD=$3 NB=$4 NT=$5 STEPS=$6 TAG=$7 DATA=$8; shift 8
case $DATA in
  ladder) ;;
  w1147) export GPIC_SHARD_SEED=1730000000 GPIC_REGION_CROP=0 ;;
  *) echo "data must be ladder or w1147"; exit 2 ;;
esac
[ $((STEPS % 128)) -eq 0 ] || { echo "steps must be a multiple of 128"; exit 2; }
WT=$WC/project/gpic-wt/gmflow_run2
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q "lam: 0.5" $WT/baselines/PixelGen/configs_h100/gmflow_train.yaml || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
ROOT=$WC/runs/gpic-speed/gmflow_k${K}_${SIZE}_${TAG}
mkdir -p $ROOT
SHA=$(git -C $WT rev-parse HEAD)
if [ -f $ROOT/commit.txt ] && [ "$(cat $ROOT/commit.txt)" != "$SHA" ]; then echo "$ROOT was started at $(cat $ROOT/commit.txt), the worktree holds $SHA"; exit 2; fi
echo $SHA > $ROOT/commit.txt
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT GMFLOW_WT=$WT
if ls $ROOT/exp_ladder_$SIZE/ms-step=$(printf %06d $STEPS)*.ckpt > /dev/null 2>&1; then
  echo "training done before"
else
  bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $STEPS --config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k $K \
    --trainer.logger.init_args.name gmflow_k${K}_${SIZE}_${TAG} --trainer.logger.init_args.id gmflow-k${K}-${SIZE}-${TAG} >> $ROOT/train.log 2>&1
  rc=$?
  echo "train rc=$rc launches=$(wc -l < $ROOT/exp_ladder_$SIZE/launches.txt 2>/dev/null)"
  [ $rc -eq 0 ] || exit $rc
fi
[ $# -eq 0 ] && exit 0
bash $WT/jobs/evals_gmflow.sh $K $SIZE $WD $NB $NT $ROOT "$@"
