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
# usage: train_gmflow.sh <K> <size> <d> <nb> <nt> <steps> <tag> <data: ladder | w1147> [--wt <worktree>] [--spread pixel] [--lam <value>] [--scope image] [step:sampler:calls ...]
# --spread pixel: train with gm_spread pixel (a per-pixel offset on log s_u, docs/gmflow.md section 7) and evaluate with it; default image. The run records its spread
# and refuses another. --wt: the frozen worktree gpic-wt/<worktree> instead of gpic-wt/gmflow_run2 (the per-pixel spread needs a commit that has it).
# --lam: the transition loss's lambda (GMFlowTrainer lam, default 0.5 as configs_h100/gmflow_train.yaml); lam 1 is the plain mixture NLL of x (docs/gmflow.md section 11).
# The run records it in lam.txt next to spread.txt (a run without lam.txt is 0.5) and refuses another value.
# --scope image: one mixture per image (gm_scope image, docs/gmflow.md section 15, owner 2026-11-06): the K means are K whole candidate images with one set of K
# weights per image; it records scope.txt (none = pixel), refuses another value, needs the image spread, and evaluates with --scope image (gm_draw, euler_mean).
# Output: runs/gpic-speed/gmflow_k<K>_<size>_<tag>/exp_ladder_<size>; W&B gmflow-k<K>-<size>-<tag>. Code: the frozen worktree gpic-wt/gmflow_run2; the run records its
# commit and refuses a worktree that moved.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
[ $# -ge 8 ] || { echo "usage: train_gmflow.sh <K> <size> <d> <nb> <nt> <steps> <tag> <ladder | w1147> [--wt <worktree>] [--spread pixel] [--lam <value>] [--scope image] [tasks...]"; exit 2; }
K=$1 SIZE=$2 WD=$3 NB=$4 NT=$5 STEPS=$6 TAG=$7 DATA=$8; shift 8
WTN=gmflow_run2
if [ "$1" = "--wt" ]; then WTN=$2; shift 2; [[ $WTN =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--wt takes a worktree name under gpic-wt/"; exit 2; }; fi
SPREAD=image
if [ "$1" = "--spread" ]; then SPREAD=$2; shift 2; case $SPREAD in image|pixel) ;; *) echo "--spread must be image or pixel"; exit 2 ;; esac; fi
LAM=0.5
if [ "$1" = "--lam" ]; then
  LAM=$2; shift 2
  [[ $LAM =~ ^(0\.[0-9]*[1-9]|1|1\.0)$ ]] || { echo "--lam must be in (0, 1], written as 0.x or 1"; exit 2; }
  [ $LAM != 1.0 ] || LAM=1
fi
SCOPE=pixel
if [ "$1" = "--scope" ]; then
  SCOPE=$2; shift 2
  case $SCOPE in pixel|image) ;; *) echo "--scope must be pixel or image"; exit 2 ;; esac
  [ $SCOPE = pixel ] || [ $SPREAD = image ] || { echo "one mixture per image takes the image spread"; exit 2; }
fi
for t in "$@"; do   # the evaluation tasks are checked before the training, so a task that the evaluation would refuse fails now, not after it (review_image_mix M2)
  [[ $t =~ ^[0-9]+:[a-z0-9_]+:[0-9]+(:[0-9]+)?$ ]] || { echo "task $t is not step:sampler:calls[:substeps]"; exit 2; }
  if [ $SCOPE = image ]; then case $t in *:gm_draw:*|*:euler_mean:*) ;; *) echo "task $t: one mixture per image is evaluated with gm_draw or euler_mean"; exit 2 ;; esac; fi
done
case $DATA in
  ladder) ;;
  w1147) export GPIC_SHARD_SEED=1730000000 GPIC_REGION_CROP=0 ;;
  *) echo "data must be ladder or w1147"; exit 2 ;;
esac
[ $((STEPS % 128)) -eq 0 ] || { echo "steps must be a multiple of 128"; exit 2; }
WT=$WC/project/gpic-wt/$WTN
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q "lam: 0.5" $WT/baselines/PixelGen/configs_h100/gmflow_train.yaml || { echo "$WT holds old code"; exit 2; }
LAM_ARGS=(); [ $LAM = 0.5 ] || LAM_ARGS=(--model.diffusion_trainer.init_args.lam $LAM)   # 0.5: the command line of before this option
SPREAD_ARGS=()
if [ $SPREAD = pixel ]; then
  grep -q "gm_spread" $WT/baselines/PixelGen/src/models/transformer/JiT_T2I.py || { echo "$WT has no per-pixel spread"; exit 2; }
  SPREAD_ARGS=(--model.denoiser.init_args.gm_spread pixel)
fi
if [ $SCOPE = image ]; then   # pixel: the command line of before this option
  grep -q "gm_scope" $WT/baselines/PixelGen/src/models/transformer/JiT_T2I.py || { echo "$WT has no gm_scope"; exit 2; }
  SPREAD_ARGS+=(--model.denoiser.init_args.gm_scope image)
fi
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
ROOT=$WC/runs/gpic-speed/gmflow_k${K}_${SIZE}_${TAG}
mkdir -p $ROOT
SHA=$(git -C $WT rev-parse HEAD)
if [ -f $ROOT/commit.txt ] && [ "$(cat $ROOT/commit.txt)" != "$SHA" ]; then echo "$ROOT was started at $(cat $ROOT/commit.txt), the worktree holds $SHA"; exit 2; fi
echo $SHA > $ROOT/commit.txt
if [ ! -f $ROOT/spread.txt ]; then   # a run started before this option has a checkpoint and no spread.txt: it is image
  if ls $ROOT/exp_ladder_$SIZE/last*.ckpt > /dev/null 2>&1; then echo image > $ROOT/spread.txt; else echo $SPREAD > $ROOT/spread.txt; fi
fi
[ "$(cat $ROOT/spread.txt)" = "$SPREAD" ] || { echo "$ROOT was started with spread $(cat $ROOT/spread.txt), not $SPREAD"; exit 2; }
if [ ! -f $ROOT/lam.txt ]; then   # a run started before this option has a checkpoint and no lam.txt: it is 0.5
  if ls $ROOT/exp_ladder_$SIZE/last*.ckpt > /dev/null 2>&1; then echo 0.5 > $ROOT/lam.txt; else echo $LAM > $ROOT/lam.txt; fi
fi
[ "$(cat $ROOT/lam.txt)" = "$LAM" ] || { echo "$ROOT was started with lam $(cat $ROOT/lam.txt), not $LAM"; exit 2; }
if [ ! -f $ROOT/scope.txt ]; then   # a run started before this option has a checkpoint and no scope.txt: it is pixel
  if ls $ROOT/exp_ladder_$SIZE/last*.ckpt > /dev/null 2>&1; then echo pixel > $ROOT/scope.txt; else echo $SCOPE > $ROOT/scope.txt; fi
fi
[ "$(cat $ROOT/scope.txt)" = "$SCOPE" ] || { echo "$ROOT was started with scope $(cat $ROOT/scope.txt), not $SCOPE"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT GMFLOW_WT=$WT
if ls $ROOT/exp_ladder_$SIZE/ms-step=$(printf %06d $STEPS)*.ckpt > /dev/null 2>&1; then
  echo "training done before"
else
  bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $STEPS --config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k $K "${SPREAD_ARGS[@]}" "${LAM_ARGS[@]}" \
    --trainer.logger.init_args.name gmflow_k${K}_${SIZE}_${TAG} --trainer.logger.init_args.id gmflow-k${K}-${SIZE}-${TAG} >> $ROOT/train.log 2>&1
  rc=$?
  echo "train rc=$rc launches=$(wc -l < $ROOT/exp_ladder_$SIZE/launches.txt 2>/dev/null)"
  [ $rc -eq 0 ] || exit $rc
fi
[ $# -eq 0 ] && exit 0
EV_OPTS=(); [ $WTN = gmflow_run2 ] || EV_OPTS+=(--wt $WTN); [ $SPREAD = image ] || EV_OPTS+=(--spread $SPREAD); [ $SCOPE = pixel ] || EV_OPTS+=(--scope $SCOPE)
bash $WT/jobs/evals_gmflow.sh $K $SIZE $WD $NB $NT $ROOT "${EV_OPTS[@]}" "$@"
