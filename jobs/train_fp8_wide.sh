#!/bin/bash
#SBATCH -J gpic-fp8-wide
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# The wide-width check of native FP8 (owner popup 2026-11-07 19:4x: 「先做宽模型检验」): the ladder's own segment (the Muon ladder
# recipe, as jobs/train_muon_ladder.sh) with GPIC_MFU_FP8=native3 (the image blocks' forward GEMMs and the MLP's dgrads in FP8,
# delayed scaling; its d640 FD arm fp8n3 passed). Everything else as the bf16 reference it is compared with, the Full ladder's
# d1280 1C0 segment w1684 (code muon_ladder_run2: lever P on, NS off). Run root runs/gpic-speed/muon_fp8check, W&B muon-fp8check-<size>.
# Code: the frozen worktree gpic-wt/muon_ladder_fp8_run (branch ladder-pns-fp8). The FP8 GPU test first.
# usage: train_fp8_wide.sh <full|lite|s80|s8> <size> <d> <nb> <nt> <C0 steps> <multiple> <adamw lr> <muon lr>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 9 ] || { echo "usage: train_muon_ladder.sh <full|lite|s80|s8> <size> <d> <nb> <nt> <C0 steps> <multiple> <adamw lr> <muon lr>"; exit 2; }
LAD=$1 SIZE=$2 WD=$3 NB=$4 NT=$5 S=$6 M=$7 ALR=$8 MLR=$9
# s80 / s8: the repetition arms (owner 2026-11-07; genmodel's repeat-arms, src/data/dataset/gpic_dataset.py EpochShuffledShards):
# the first 80 or 8 shards with the Full ladder's seed; fewer shards than 8 GPUs x 32 workers, so every pass reads k copies
case $LAD in full) PAT='gpic_train_{00000..07999}.tar' SEED=20261106 ;; lite) PAT='gpic_train_{00000..00800}.tar' SEED=20261107 ;;
  s80) PAT='gpic_train_{00000..00079}.tar' SEED=20261106 ;; s8) PAT='gpic_train_{00000..00007}.tar' SEED=20261106 ;;
  *) echo "ladder must be full, lite, s80 or s8"; exit 2 ;; esac
case $M in 1|2|4|8) ;; *) echo "multiple must be 1, 2, 4 or 8"; exit 2 ;; esac
[ $((S % 32)) -eq 0 ] || { echo "C0 steps must be a multiple of 32"; exit 2; }
[ "$SIZE" = "d$WD" ] || { echo "size $SIZE does not match d $WD"; exit 2; }
WT=$WC/project/gpic-wt/muon_ladder_fp8_run   # ladder-pns-fp8: 3338a8c + P + NS (off here) + native FP8
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "update_scale: mup" $WT/baselines/PixelGen/configs_h100/muon_ladder.yaml || { echo "$WT has no Muon ladder config"; exit 2; }
(cd $WT && CUDA_VISIBLE_DEVICES=0 $WC/envs/pixgen/bin/python tools/mfu/tests/test_fp8_native.py 2>&1 | grep -v -i warn | tail -12) | tee /dev/stderr | grep -q "FP8_NATIVE_TEST PASS" || { echo "FP8 test failed"; exit 1; }
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
export GPIC_MFU_FSDP=$FS GPIC_MFU_FSDP_GROUP=$G GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=$FS GPIC_MFU_NS_FUSED=0 GPIC_MFU_FP8=native3   # as w1684 (P), plus FP8
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_fp8check LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $((S * M)) \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((1024 / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" \
  --trainer.logger.init_args.name muon_fp8check_$SIZE --trainer.logger.init_args.id muon-fp8check-$SIZE
