#!/bin/bash
#SBATCH -J gpic-bucket-ab-fp8n3
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# The FD arm of native FP8 phase 3 (GPIC_MFU_FP8=native3: the image blocks' four GEMMs in FP8 forward, the MLP's dgrads
# in FP8, delayed scaling for the activation casts; owner 2026-11-07: 「FP8要做」, 「直接做b」; FP8 changes the numerics):
# identical to genmodel's shared-bucket arm shared128 (muon-search2 jobs/train_bucket_ab2.sh shared128 true 128 33408 on:
# width 640, Muon 3.4e-3, AdamW 6.4e-4, batch 1024 as 128 x 1 on 8 GPUs, GPIC Full, seed 20261106, shared bucket, kernels
# K and K2 on, 33408 steps) except the FP8 GEMMs. Root runs/gpic-speed/muon_bucket_ab/fp8n3 (heldout_bucket_ab.sh picks it
# up). Code: the frozen worktree gpic-wt/mfu_fp8_run (branch mfu: the ladder's code 3338a8c plus switches that are off here,
# and native FP8). Targets at 33408: FD shared 1016.6 (seed 20261106), 966.6 (shared128-s1).
# usage: train_fp8_fd.sh
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
TAG=fp8n3 S=33408 MBX=128
SIZE=d640 WD=640 NB=9 NT=2 ALR=6.4e-4 MLR=3.4e-3
PAT='gpic_train_{00000..07999}.tar' SEED=20261106
WT=$WC/project/gpic-wt/mfu_fp8_run
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "class _Fp8MLP3" $WT/baselines/PixelGen/src/models/layers/fp8_native.py || { echo "$WT has no FP8 phase 3"; exit 2; }
# the FP8 kernels on this GPU first (casts bit for bit, delayed scaling, a block eager against compiled)
(cd $WT && CUDA_VISIBLE_DEVICES=0 $WC/envs/pixgen/bin/python tools/mfu/tests/test_fp8_native.py 2>&1 | grep -v -i warn | tail -12) | tee /dev/stderr | grep -q "FP8_NATIVE_TEST PASS" || { echo "FP8 test failed"; exit 1; }
export GPIC_MFU_FSDP=0 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_FP8=native3
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_bucket_ab/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
echo "$(date -u +%FT%TZ) GPIC_MFU_FP8=$GPIC_MFU_FP8 code $(git -C $WT rev-parse --short HEAD)" >> $LADDER_RUN_ROOT/fp8.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --data.train_batch_size $MBX --trainer.accumulate_grad_batches 1 --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" \
  --trainer.logger.init_args.name bucket_ab_${TAG}_$SIZE --trainer.logger.init_args.id bucket-ab-$TAG-$SIZE
