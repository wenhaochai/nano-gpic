#!/bin/bash
#SBATCH -J gpic-fp8branch
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# FP8 branch runs (gpic-28; owner 2026-11-08 「可以」): where does FP8 lose the d16 check (FD +9.9% at C0; held-out -0.09% at 10200 but
# +0.10% at 13600)? The check's last 3400 steps (10200 -> 13600: 680 steps at the peak rate, then the linear cooldown over 2720 steps to
# 3e-5) re-run from either arm's 10200 weights (raw; GPIC_INIT_CKPT, a fresh optimizer with a 1% warmup), each in bf16 or FP8 native3,
# all on one data order (shard seed) and one torch seed except the noise replicate. Otherwise genmodel's recipe as in train_sweep_fp8.sh:
# d16 (width 1536, 16 + 4 blocks), FSDP2 G1, 128 x 1, batch 1024, Muon 6.8e-3, AdamW 0.000179, no weight decay, no EMA, NVLS off.
# Root runs/gpic-speed/muon_fp8branch/<tag>.
# usage: train_fp8_branch.sh <tag> <start checkpoint> <fp8: none|native3> <shard seed> <torch seed>
export GPIC_MFU_NO_EMA=1   # owner 2026-11-08 「ema是一定不能用」
export NCCL_NVLS_ENABLE=0  # NCCL 2.26.2 NVLS all-gather can permute rank blocks (NVIDIA/nccl #1906)
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 5 ] || { echo "usage: train_fp8_branch.sh <tag> <start checkpoint> <fp8: none|native3> <shard seed> <torch seed>"; exit 2; }
TAG=$1 INIT=$2 F8=$3 SSEED=$4 TSEED=$5
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
[ -f "$INIT" ] || { echo "no $INIT"; exit 2; }
[[ $TAG =~ ^[0-9a-z.-]+$ && $SSEED =~ ^[0-9]+$ && $TSEED =~ ^[0-9]+$ ]] || { echo "bad tag or seed"; exit 2; }
WD=1536 NB=16 NT=4 S=3400 MLR=6.8e-3 ALR=0.000179 WDK=0 GB=1024 CD=2720 WU=34 MB=128
SIZE=d$WD PAT='gpic_train_{00000..07999}.tar'
WT=$WC/project/gpic-wt/fp8_branch_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
[ -f $WT/baselines/PixelGen/src/models/layers/fp8_native.py ] || { echo "$WT has no native FP8"; exit 2; }
export GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=1 GPIC_MFU_MUON_ADD=1 GPIC_MFU_FP8=$F8
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_fp8branch/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SSEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
[ -f $LADDER_RUN_ROOT/exp_ladder_$SIZE/last.ckpt ] || export GPIC_INIT_CKPT=$INIT   # the start weights only for a fresh run; a resume restores its own state
echo "branch $TAG: start $INIT, fp8 [${F8:-none}], shard seed $SSEED, torch seed $TSEED, init [${GPIC_INIT_CKPT:-resume}]"
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" --seed_everything $TSEED \
  --trainer.logger.init_args.name fp8branch_${TAG}_$SIZE --trainer.logger.init_args.id fp8branch-$TAG-$SIZE
