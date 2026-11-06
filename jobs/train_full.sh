#!/bin/bash
#SBATCH -J gpic-full
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# One budget segment of a full-ladder run (docs/ladder_plan.md section 12, owner 2026-11-06): the ladder's point model with
# the Lite ladder's recipe (configs_h100/lite_ladder.yaml) on GPIC Full (shards 00000-07999 and their text cache), width d,
# trained to <multiple> x <C0 steps> steps. All segments of a size share the run directory
# runs/gpic-speed/full_ladder/exp_ladder_<size> and the W&B run ladder-full-<size>: segment k resumes the run where segment
# k-1 stopped (flat LR after the warm-up), with the same shard order (jobs/train_ladder.sh). ms checkpoints fall at the
# multiples of steps/4: 1 x C0 -> C0/4, C0/2, 3C0/4, C0; 2 -> 1.5C0, 2C0; 4 -> 3C0, 4C0; 8 -> 6C0, 8C0. last.ckpt every
# C0/8 steps, so a stop loses at most C0/8 of training and a size saves its full state 8 times per C0.
# Code: the frozen worktree gpic-wt/ladder_full_run. Evaluations: gpic-wt/gmflow_run4/jobs/evals_gmflow.sh (official noise).
# usage: train_full.sh <size> <d> <nb> <nt> <C0 steps> <multiple>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 6 ] || { echo "usage: train_full.sh <size> <d> <nb> <nt> <C0 steps> <multiple>"; exit 2; }
SIZE=$1 WD=$2 NB=$3 NT=$4 S=$5 M=$6
case $M in 1|2|4|8|16) ;; *) echo "multiple must be 1, 2, 4, 8 or 16"; exit 2 ;; esac
[ $((S % 800)) -eq 0 ] || { echo "C0 steps must be a multiple of 800"; exit 2; }
WT=${FULL_WT:-$WC/project/gpic-wt/ladder_full_run}   # FULL_WT: smoke tests only
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=${FULL_ROOT:-$WC/runs/gpic-speed/full_ladder} LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $((S * M)) \
  --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
  --trainer.logger.init_args.name ladder_full_$SIZE --trainer.logger.init_args.id ladder-full-$SIZE
