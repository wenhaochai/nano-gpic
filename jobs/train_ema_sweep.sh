#!/bin/bash
#SBATCH -J gpic-ema-sweep
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# One budget segment of the new ladders (owner 2026-11-07 01:2x: 「第一阶段结束之后...起两个ladder（一个full一个lite）」; started early at
# the queue-emergency rule): the Muon ladder recipe (configs_h100/muon_ladder.yaml, global batch 1024 on 8 GPUs, split per width below) on GPIC Full
# (shards 00000-07999) or Lite (00000-00800), width d, trained to <multiple> x <C0 steps> optimizer steps (C0 = 1.43e19 FLOPs at
# batch 1024, a multiple of 32). All segments of a ladder and size share runs/gpic-speed/muon_<ladder>/exp_ladder_<size> and the
# W&B run muon-<ladder>-<size>; a later segment continues the run (flat lr after warmup). ms checkpoints at the multiples of
# steps/4; last.ckpt every C0/8. One shard seed per ladder for every width. Code: the frozen worktree gpic-wt/muon_ladder_run.
# EMA sweep (owner 2026-11-07 10:1x popup: 「扫一下EMA 比如搞成0.9999看看」): the Full ladder's recipe (Muon 3.4e-3, batch 1024, GPIC Full, seed
# 20261106) at one width to <steps>, with the EMA decay as an argument (the ladder uses 0.9996 per optimizer step). Root
# runs/gpic-speed/muon_ema_sweep/<tag>/exp_ladder_<size>; the 0.9996 arm is the Full ladder's own run (runs/gpic-speed/muon_full).
# usage: train_ema_sweep.sh <size> <d> <nb> <nt> <steps> <adamw lr> <muon lr> <ema decay> <tag>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 9 ] || { echo "usage: train_ema_sweep.sh <size> <d> <nb> <nt> <steps> <adamw lr> <muon lr> <ema decay> <tag>"; exit 2; }
SIZE=$1 WD=$2 NB=$3 NT=$4 S=$5 ALR=$6 MLR=$7 EMA=$8 TAG=$9 M=1 LAD=ema-sweep
[[ $EMA =~ ^0\.9[0-9]+$ ]] || { echo "ema decay must look like 0.9999"; exit 2; }
[[ $TAG =~ ^[0-9a-z-]+$ ]] || { echo "bad tag"; exit 2; }
# s80 / s8: the repetition arms (owner 2026-11-07; genmodel's repeat-arms, src/data/dataset/gpic_dataset.py EpochShuffledShards):
# the first 80 or 8 shards with the Full ladder's seed; fewer shards than 8 GPUs x 32 workers, so every pass reads k copies
PAT='gpic_train_{00000..07999}.tar' SEED=20261106
[ $((S % 32)) -eq 0 ] || { echo "steps must be a multiple of 32"; exit 2; }
[ "$SIZE" = "d$WD" ] || { echo "size $SIZE does not match d $WD"; exit 2; }
WT=$WC/project/gpic-wt/muon_ladder_run   # the frozen final code (ddd4b99)
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "update_scale: mup" $WT/baselines/PixelGen/configs_h100/muon_ladder.yaml || { echo "$WT has no Muon ladder config"; exit 2; }
# Speed settings per width (gpic-28, docs/mfu.md "Batch 1024 per width"; w1592 / w1596, 8 GPUs, Muon): the same math as
# 8 x 32 x 4 (the gradient is the mean over the same 1024 samples), without accumulation where 128 per GPU fits (owner
# 2026-11-07: 「累积步数能不开尽可能就是不开的」); the shared caption bucket (owner's popup 01:3x); levers A C D K K2 at every
# width; FSDP2 (B, no-sync under accumulation) where it beats DDP. Measured ms per optimizer step (mean, second half; MFU):
# d640 76.5 (17.6%), d768 83.6 (25.4%), d896 103.9 (30.2%), d1024 128.9 (34.8%), d1280 211.2 (38.1%; DDP 213.5),
# d1536 324.0 (40.9%; 64 x 2: 333.9, 39.7%), d1792 64 x 2: see docs/mfu.md. Peak reserved memory at 128 x 1: d1536 70.5 GiB
# of 79.6 (the warm-up's T = 320 batch fits in it); 128 at d1792 does not fit.
case $WD in
  640|768|896|1024) MB=128 FS=0 G=1 ;;
  1280)             MB=128 FS=1 G=7 ;;
  1536)             MB=128 FS=1 G=1 ;;
  1792)             MB=64 FS=1 G=1 ;;
  *) echo "no speed settings for d$WD"; exit 2 ;;
esac
export GPIC_MFU_FSDP=$FS GPIC_MFU_FSDP_GROUP=$G GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_ema_sweep/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $((S * M)) \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR --model.ema_tracker.init_args.decay $EMA \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((1024 / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" \
  --trainer.logger.init_args.name ema_sweep_${TAG}_$SIZE --trainer.logger.init_args.id ema-sweep-$TAG-$SIZE
