#!/bin/bash
#SBATCH -J gpic-gmflow-evals
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# A list of evaluations of one run (docs/gmflow.md section 6): jobs/eval_gmflow.sh for each <step>:<sampler>:<calls>, at any width; 50K test captions, each
# prompt's own noise, EMA, no CFG. Finished tasks are skipped, so a rerun does only the rest.
# K = 0 evaluates a point model. --link <run dir> links that run's ms checkpoints into <root>/exp_ladder_<size>, so nothing is written in its directory (the ladder's
# own evaluations use one noise for all prompts, so the ladder's checkpoints are evaluated again here).
# usage: evals_gmflow.sh <K> <size> <d> <nb> <nt> <root> [--link <run dir>] [--wt <worktree>] [--spread pixel] <step:sampler:calls> ...
# --spread pixel: the run has the per-pixel spread (docs/gmflow.md section 7; eval_gmflow.sh EVAL_SPREAD); default image.
# Code: the frozen worktree gpic-wt/gmflow_run2, or gpic-wt/<worktree> with --wt (the per-pixel spread and gm_sde2 / gm_ode2 need a commit that has them; a wq job
# cannot take it from the environment); every evaluation checks that it still holds the commit of the start.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
[ $# -ge 7 ] || { echo "usage: evals_gmflow.sh <K> <size> <d> <nb> <nt> <root> [--link <dir>] [--wt <worktree>] [--spread pixel] <tasks...>"; exit 2; }
K=$1 SIZE=$2 WD=$3 NB=$4 NT=$5 ROOT=$6; shift 6
LINK=""
if [ "$1" = "--link" ]; then LINK=$2; shift 2; [ -d "$LINK" ] || { echo "no run directory $LINK"; exit 2; }; fi
WTN=gmflow_run2
if [ "$1" = "--wt" ]; then WTN=$2; shift 2; fi
SPREAD=image
if [ "$1" = "--spread" ]; then SPREAD=$2; shift 2; case $SPREAD in image|pixel) ;; *) echo "--spread must be image or pixel"; exit 2 ;; esac; fi
[ $# -ge 1 ] || { echo "no tasks"; exit 2; }
WT=$WC/project/gpic-wt/$WTN
[ -f "$WT/baselines/PixelGen/src/diffusion/flow_matching/sampling_gmflow.py" ] || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SHA=$(git -C $WT rev-parse HEAD)
RUN=$ROOT/exp_ladder_$SIZE
mkdir -p $RUN
for t in "$@"; do
  case $t in *:*:*) ;; *) echo "task $t is not step:sampler:calls"; exit 2 ;; esac
  S6=$(printf %06d ${t%%:*})
  if [ -n "$LINK" ] && [ ! -e $RUN/ms-step=$S6.ckpt ]; then
    [ -f $LINK/ms-step=$S6.ckpt ] || { echo "no $LINK/ms-step=$S6.ckpt"; exit 2; }
    ln -s $LINK/ms-step=$S6.ckpt $RUN/
  fi
done
[ -f $ROOT/commit.txt ] || echo $SHA > $ROOT/commit.txt
export GMFLOW_WT=$WT LADDER_RUN_ROOT=$ROOT EVAL_SIZE=$SIZE EVAL_D=$WD EVAL_NB=$NB EVAL_NT=$NT EVAL_SPREAD=$SPREAD
fail=0
for t in "$@"; do
  STEP=${t%%:*} rest=${t#*:}; S=${rest%%:*} C=${rest#*:}
  if [ -f $RUN/eval_$(printf %06d $STEP)_${S}_calls$C/metrics.json ]; then echo "eval $t done before"; continue; fi
  [ "$(git -C $WT rev-parse HEAD)" = "$SHA" ] || { echo "$WT moved from $SHA"; exit 2; }
  bash $WT/jobs/eval_gmflow.sh $K $STEP $S $C > $ROOT/eval_${STEP}_${S}_$C.log 2>&1 || fail=1
  echo "eval $t fail=$fail"
done
exit $fail
