#!/bin/bash
#SBATCH -J gpic-bucket-ab-gdec128
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# The FD arm of GPU JPEG decoding (GPIC_MFU_GPU_DECODE=2; owner 2026-11-07: 「可以用GPU解码」; nvJPEG's pixels differ slightly from
# libjpeg-turbo's, so the data changes): identical to genmodel's shared-bucket arm shared128 (muon-search2 jobs/train_bucket_ab2.sh
# shared128 true 128 33408 on: width 640, Muon 3.4e-3, AdamW 6.4e-4, batch 1024 as 128 x 1 on 8 GPUs, GPIC Full, seed 20261106,
# shared bucket, kernels K and K2 on, 33408 steps) except that about 84% of the training images are decoded by nvJPEG in the step,
# a step ahead in a background thread. Root runs/gpic-speed/muon_bucket_ab/gdec128 (heldout_bucket_ab.sh picks it up). Code: the
# frozen worktree gpic-wt/mfu_gdec_run (branch mfu: the ladder's code 3338a8c plus switches that are off here, and GPU decoding).
# Targets at 33408: FD shared 1016.6 (seed 20261106), 966.6 (shared128-s1).
# usage: train_gdec_fd.sh
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
TAG=gdec128 S=33408 MBX=128
SIZE=d640 WD=640 NB=9 NT=2 ALR=6.4e-4 MLR=3.4e-3
PAT='gpic_train_{00000..07999}.tar' SEED=20261106
WT=$WC/project/gpic-wt/mfu_gdec_run
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "class DecodeAhead" $WT/baselines/PixelGen/src/data/dataset/pil_exact_gpu.py || { echo "$WT has no GPU decoding"; exit 2; }
# the decoding plumbing on this GPU first (bit for bit against the CPU path with Pillow's pixels; DecodeAhead; the GIL)
(cd $WT && GPIC_REGION_CROP=1 CUDA_VISIBLE_DEVICES=0 $WC/envs/pixgen/bin/python tools/mfu/tests/test_gpu_decode.py 2>&1 | grep -v -i warn | tail -9) | tee /dev/stderr | grep -q "GPU_DECODE_TEST PASS" || { echo "GPU decoding test failed"; exit 1; }
export GPIC_MFU_FSDP=0 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_GPU_DECODE=2
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_bucket_ab/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
echo "$(date -u +%FT%TZ) GPIC_MFU_GPU_DECODE=$GPIC_MFU_GPU_DECODE code $(git -C $WT rev-parse --short HEAD)" >> $LADDER_RUN_ROOT/gpu_decode.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --data.train_batch_size $MBX --trainer.accumulate_grad_batches 1 --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" \
  --trainer.logger.init_args.name bucket_ab_${TAG}_$SIZE --trainer.logger.init_args.id bucket-ab-$TAG-$SIZE
