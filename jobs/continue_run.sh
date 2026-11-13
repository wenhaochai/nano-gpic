#!/bin/bash
#SBATCH -J gpic-continue
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=04:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# Continue a finished run to more steps (docs/gmflow.md section 8, group M, owner 2026-11-05). The LR is flat after the warm-up and EMA, optimizer and step
# counter come back from last.ckpt, so a run continued from its last checkpoint to 2 STEPS is a run trained to 2 STEPS; a new launch draws a new shard order,
# as every resumed ladder run does. The source's last.ckpt is copied, not linked, into <new root>/exp_ladder_<size>, so a save cannot reach the source run.
# K = 0: a ladder point model, with the ladder's frozen code gpic-wt/ladder_run; K > 0: GMFlow, with gpic-wt/gmflow_b2 (gmflow_run2's code plus these scripts).
# Resumable (wq --partial): train_ladder.sh resumes from the newest last*.ckpt and exits 0 only when the final ms checkpoint exists.
# usage: continue_run.sh <K> <size> <d> <nb> <nt> <source run dir> <new root name> <steps> <wandb name>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
[ $# -eq 9 ] || { echo "usage: continue_run.sh <K> <size> <d> <nb> <nt> <source run dir> <new root name> <steps> <wandb name>"; exit 2; }
K=$1 SIZE=$2 WD=$3 NB=$4 NT=$5 SRC=$6 NAME=$7 STEPS=$8 WB=$9
case $K in
  0) WT=$WC/project/gpic-wt/ladder_run; EXTRA=() ;;
  [1-9]|[1-9][0-9]) WT=$WC/project/gpic-wt/gmflow_b2; EXTRA=(--config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k $K) ;;
  *) echo "K must be 0 to 99"; exit 2 ;;
esac
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
case $NAME in ""|*/*) echo "bad root name $NAME"; exit 2 ;; esac
ROOT=$WC/runs/gpic-speed/$NAME
RUN=$ROOT/exp_ladder_$SIZE
mkdir -p $RUN
SHA=$(git -C $WT rev-parse HEAD)
if [ -f $ROOT/commit.txt ] && [ "$(cat $ROOT/commit.txt)" != "$SHA" ]; then echo "$ROOT was started at $(cat $ROOT/commit.txt), the worktree holds $SHA"; exit 2; fi
echo $SHA > $ROOT/commit.txt
if ls $RUN/ms-step=$(printf %06d $STEPS)*.ckpt > /dev/null 2>&1; then echo "training done before"; exit 0; fi
if ! ls $RUN/last*.ckpt > /dev/null 2>&1; then
  [ -f $SRC/last.ckpt ] || { echo "no $SRC/last.ckpt"; exit 2; }
  cp $SRC/last.ckpt $RUN/last.ckpt.tmp && mv $RUN/last.ckpt.tmp $RUN/last.ckpt || { echo "copy of $SRC/last.ckpt failed"; exit 2; }
  echo "$(date -u +%FT%TZ) copied $SRC/last.ckpt ($(stat -c %y $SRC/last.ckpt))" | tee -a $ROOT/source.txt
fi
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT
bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $STEPS "${EXTRA[@]}" \
  --trainer.logger.init_args.name $WB --trainer.logger.init_args.id ${WB//_/-} >> $ROOT/train.log 2>&1
rc=$?
echo "train rc=$rc launches=$(wc -l < $RUN/launches.txt 2>/dev/null)"
exit $rc
