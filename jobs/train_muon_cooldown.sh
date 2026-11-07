#!/bin/bash
#SBATCH -J gpic-muon-cooldown
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# A learning-rate cooldown in place of the EMA (owner 2026-11-07: 「ema的替代实验」, popup: widths 640 and 1024, from scratch to C0,
# 「学习率线性不降到0，而是降低到一个固定值3e-5」): the Full ladder's recipe and data order exactly (configs_h100/muon_ladder.yaml, 128 x 1,
# shared bucket, EMA 0.9996, warmup 1000, seed_everything 0, shard seed 20261106), except that the rate stays flat until
# <steps> - <cooldown steps> and then falls linearly to the absolute rate <min lr> in every param group at <steps>
# (src/utils/lr_scheduler.py WarmupFlatCooldown min_lr on branch cooldown; tools/test_cooldown_min_lr.py). Compare the raw
# weights at <steps> with the main line's EMA and raw weights at the same step. Root runs/gpic-speed/<root>/<tag>/exp_ladder_<size>.
# usage: train_muon_cooldown.sh <root> <tag> <size> <d> <nb> <nt> <steps> <cooldown steps> <min lr> <adamw lr> <muon lr>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 11 ] || { echo "usage: train_muon_cooldown.sh <root> <tag> <size> <d> <nb> <nt> <steps> <cooldown steps> <min lr> <adamw lr> <muon lr>"; exit 2; }
ROOTN=$1 TAG=$2 SIZE=$3 WD=$4 NB=$5 NT=$6 S=$7 CD=$8 MINLR=$9 ALR=${10} MLR=${11} M=1 LAD=cooldown
MBX=128 ACC=1 SB=true EMA=0.9996 WU=1000 SEEDE=0   # the Full ladder's values
[[ $CD =~ ^[0-9]+$ && $CD -gt 0 && $CD -lt $S && $MINLR =~ ^[0-9.e-]+$ ]] || { echo "bad cooldown steps or min lr"; exit 2; }
[[ $ROOTN =~ ^muon_[a-z0-9_]+$ && $TAG =~ ^[0-9a-z-]+$ ]] || { echo "bad root or tag"; exit 2; }
case $SB in true|false) ;; *) echo "shared bucket must be true or false"; exit 2 ;; esac
[[ $MBX =~ ^(32|64|128)$ && $ACC =~ ^[1-8]$ && $WU =~ ^[0-9]+$ && $SEEDE =~ ^[0-9]+$ && $EMA =~ ^0\.9[0-9]+$ ]] || { echo "bad micro-batch, accumulation, warmup, seed or EMA"; exit 2; }
[[ $TAG =~ ^[0-9a-z-]+$ ]] || { echo "bad tag"; exit 2; }
# s80 / s8: the repetition arms (owner 2026-11-07; genmodel's repeat-arms, src/data/dataset/gpic_dataset.py EpochShuffledShards):
# the first 80 or 8 shards with the Full ladder's seed; fewer shards than 8 GPUs x 32 workers, so every pass reads k copies
PAT='gpic_train_{00000..07999}.tar' SEED=20261106
[ $((S % 32)) -eq 0 ] || { echo "steps must be a multiple of 32"; exit 2; }
[ "$SIZE" = "d$WD" ] || { echo "size $SIZE does not match d $WD"; exit 2; }
WT=$WC/project/gpic-wt/cooldown_run   # the ladder code (3338a8c) plus min_lr in WarmupFlatCooldown (branch cooldown)
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
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/$ROOTN/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $((S * M)) \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR --model.ema_tracker.init_args.decay $EMA --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr $MINLR --seed_everything $SEEDE \
  --data.train_batch_size $MBX --trainer.accumulate_grad_batches $ACC --data.train_dataset.init_args.shared_bucket $SB \
  --data.train_dataset.init_args.tar_pattern "$PAT" \
  --trainer.logger.init_args.name ${ROOTN}_${TAG}_$SIZE --trainer.logger.init_args.id ${ROOTN//_/-}-$TAG-$SIZE
