#!/bin/bash
#SBATCH -J gpic-muon-ladder
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# One budget segment of the new ladders (owner 2026-11-07 01:2x: 「第一阶段结束之后...起两个ladder（一个full一个lite）」; started early at
# the queue-emergency rule): the Muon ladder recipe (configs_h100/muon_ladder.yaml, batch 1024 = 8 x 32 x accumulation 4) on GPIC Full
# (shards 00000-07999) or Lite (00000-00800), width d, trained to <multiple> x <C0 steps> optimizer steps (C0 = 1.43e19 FLOPs at
# batch 1024, a multiple of 32). All segments of a ladder and size share runs/gpic-speed/muon_<ladder>/exp_ladder_<size> and the
# W&B run muon-<ladder>-<size>; a later segment continues the run (flat lr after warmup). ms checkpoints at the multiples of
# steps/4; last.ckpt every C0/8. One shard seed per ladder for every width. Code: the frozen worktree gpic-wt/muon_ladder_run.
# usage: train_muon_ladder.sh <full|lite> <size> <d> <nb> <nt> <C0 steps> <multiple> <adamw lr> <muon lr>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 9 ] || { echo "usage: train_muon_ladder.sh <full|lite> <size> <d> <nb> <nt> <C0 steps> <multiple> <adamw lr> <muon lr>"; exit 2; }
LAD=$1 SIZE=$2 WD=$3 NB=$4 NT=$5 S=$6 M=$7 ALR=$8 MLR=$9
case $LAD in full) PAT='gpic_train_{00000..07999}.tar' SEED=20261106 ;; lite) PAT='gpic_train_{00000..00800}.tar' SEED=20261107 ;; *) echo "ladder must be full or lite"; exit 2 ;; esac
case $M in 1|2|4|8) ;; *) echo "multiple must be 1, 2, 4 or 8"; exit 2 ;; esac
[ $((S % 32)) -eq 0 ] || { echo "C0 steps must be a multiple of 32"; exit 2; }
[ "$SIZE" = "d$WD" ] || { echo "size $SIZE does not match d $WD"; exit 2; }
WT=${LADDER_CODE_WT:-$WC/project/gpic-wt/muon_ladder_run}
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "update_scale: mup" $WT/baselines/PixelGen/configs_h100/muon_ladder.yaml || { echo "$WT has no Muon ladder config"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_$LAD LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $((S * M)) \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --data.train_dataset.init_args.tar_pattern "$PAT" \
  --trainer.logger.init_args.name muon_${LAD}_$SIZE --trainer.logger.init_args.id muon-$LAD-$SIZE
