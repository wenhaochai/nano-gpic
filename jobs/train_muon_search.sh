#!/bin/bash
#SBATCH -J gpic-muon-search
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# Muon learning-rate search (owner popup 2026-11-07 01:0x): the Muon ladder recipe of configs_h100/muon_ladder.yaml (MuP-scaled
# Muon on the blocks' split matrices, Nesterov 0.95, 5 Newton-Schulz steps, wd 0; AdamW 0.9/0.95 for the rest at the width's lr;
# warmup 1000 optimizer steps, then flat; batch 1024 = 8 GPUs x 32 x accumulation 4; EMA 0.9996) on GPIC Full (shards
# 00000-07999, the Full ladder's shard seed 20261106), width d, to <steps> optimizer steps = C0/4 = 3.6e18 FLOPs at batch 1024,
# rounded to a multiple of 32 (d640 33408, d1024 10048). ms checkpoints at steps/4, /2, 3/4 and the end (jobs/train_ladder.sh).
# Root runs/gpic-speed/muon_lr_search/mlr<tag>/exp_ladder_<size>; W&B muon-search-mlr<tag>-<size>.
# Code: the frozen worktree gpic-wt/muon_search_run (branch muon-search from mfu@9928ffe, gpic-28's Muon, GPU test w1484).
# usage: train_muon_search.sh <size> <d> <nb> <nt> <steps> <adamw lr> <muon lr> <tag>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 8 ] || { echo "usage: train_muon_search.sh <size> <d> <nb> <nt> <steps> <adamw lr> <muon lr> <tag>"; exit 2; }
SIZE=$1 WD=$2 NB=$3 NT=$4 S=$5 ALR=$6 MLR=$7 TAG=$8
[[ $TAG =~ ^[0-9a-z-]+$ ]] || { echo "tag must be lowercase letters, digits and dashes"; exit 2; }
[ "$SIZE" = "d$WD" ] || { echo "size $SIZE does not match d $WD"; exit 2; }
WT=${SEARCH_WT:-$WC/project/gpic-wt/muon_search_run}   # SEARCH_WT: smoke tests only
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "update_scale: mup" $WT/baselines/PixelGen/configs_h100/muon_ladder.yaml || { echo "$WT has no Muon ladder config"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=${SEARCH_ROOT:-$WC/runs/gpic-speed/muon_lr_search/mlr$TAG} LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo 20261106 > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..07999}.tar' \
  --trainer.logger.init_args.name muon_search_mlr${TAG}_$SIZE --trainer.logger.init_args.id muon-search-mlr$TAG-$SIZE
