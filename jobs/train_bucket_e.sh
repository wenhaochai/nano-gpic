#!/bin/bash
#SBATCH -J gpic-bucket-ab-e128
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# The FD arm of packed captions (lever E with K2 on the packed layout; owner 2026-11-07 12:3x: 「先测 FD 再切换」, 「插队这个测试」):
# identical to genmodel's bucket A/B arms (muon-search2 jobs/train_bucket_ab2.sh: width 640, Muon 3.4e-3, AdamW 6.4e-4, batch 1024 as
# 128 x 1 on 8 GPUs, GPIC Full, seed 20261106, warmup 1000, EMA 0.9996, kernels K and K2 on, 33408 steps) except the batches: random
# (group_window = the micro-batch, no length grouping, no shared bucket) with the captions packed (GPIC_MFU_PACK=1), so every optimizer
# step mixes all caption lengths. Root runs/gpic-speed/muon_bucket_ab/e128 (heldout_bucket_ab.sh picks it up). Code: the frozen worktree
# gpic-wt/mfu_e128_run2 (branch mfu with the packed K2 and its NaN fix f3d78ad; the first try w1871 on 042541f trained to NaN). Targets at 33408: FD shared 1016.6, rank128 982.6, rank32 959.3.
# usage: train_bucket_e.sh
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
TAG=e128 S=33408 MBX=128
SIZE=d640 WD=640 NB=9 NT=2 ALR=6.4e-4 MLR=3.4e-3
PAT='gpic_train_{00000..07999}.tar' SEED=20261106
WT=$WC/project/gpic-wt/mfu_e128_run2
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "def qk_prep_packed" $WT/baselines/PixelGen/src/models/layers/fused_qk.py || { echo "$WT has no packed K2"; exit 2; }
# the packed K2 on this GPU first (FA3 end to end, NaN-poisoned memory): a NaN run costs 45 min of 8 GPUs
(cd $WT && CUDA_VISIBLE_DEVICES=0 $WC/envs/pixgen/bin/python tools/mfu/tests/test_fused_qk_packed.py 2>&1 | tail -4) | tee /dev/stderr | grep -q "FUSED_QK_PACKED_TEST PASS" || { echo "packed K2 test failed"; exit 1; }
export GPIC_MFU_FSDP=0 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=0
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_bucket_ab/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --data.train_batch_size $MBX --trainer.accumulate_grad_batches 1 --data.train_dataset.init_args.shared_bucket false \
  --data.train_dataset.init_args.group_window $MBX --data.train_dataset.init_args.tar_pattern "$PAT" \
  --trainer.logger.init_args.name bucket_ab_${TAG}_$SIZE --trainer.logger.init_args.id bucket-ab-$TAG-$SIZE
