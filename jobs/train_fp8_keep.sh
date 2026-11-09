#!/bin/bash
#SBATCH -J gpic-fp8keep
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The resume test of the FP8 outlier (owner popup 2026-11-08 23:4x, via genmodel: 「先查清离群点再定」): the first FP8 run resumed at 11,900, inside
# the cooldown (from 10,880), and read held-out +0.1% and FD +10% at C0; the one-piece replicate did not. This continuous arm repeats the
# one-piece FP8 replicate (w2327's seeds) in one piece to 13,600 and keeps a full-state copy at 11,904 (last-keep-step=011904.ckpt;
# CheckpointHook keeps the full state only in last*.ckpt), from which jobs/resume_fp8_keep.sh resumes the paired arm. Per-step losses
# in <root>/bench (BENCH_OUT). Otherwise train_fp8_rep_once.sh.
# usage: train_fp8_keep.sh <tag> <fp8: none|native3> <shard seed> <torch seed>
export GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 4 ] || { echo "usage: train_fp8_keep.sh <tag> <fp8: none|native3> <shard seed> <torch seed>"; exit 2; }
TAG=$1 F8=$2 SSEED=$3 TSEED=$4
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
[[ $TAG =~ ^[0-9a-z.-]+$ && $SSEED =~ ^[0-9]+$ && $TSEED =~ ^[0-9]+$ ]] || { echo "bad tag or seed"; exit 2; }
WD=1536 NB=16 NT=4 S=13600 MLR=6.8e-3 ALR=0.000179 WDK=0 GB=1024 CD=2720 WU=136 MB=128
SIZE=d$WD PAT='gpic_train_{00000..07999}.tar'
WT=$WC/project/gpic-wt/fp8_branch_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
unset GPIC_INIT_CKPT GPIC_DEBUG_OPT_STATE
export GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=1 GPIC_MFU_MUON_ADD=1 GPIC_MFU_FP8=$F8
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_fp8recipe/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ ! -f $LADDER_RUN_ROOT/exp_ladder_$SIZE/last.ckpt ] || { echo "$LADDER_RUN_ROOT has a last.ckpt: this run must be in one piece"; exit 3; }
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SSEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
export BENCH_OUT=$LADDER_RUN_ROOT/bench
KEEP='{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"last-keep-{step:06d}","monitor":null,"save_last":false,"save_top_k":-1,"every_n_train_steps":11904}}'
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" --seed_everything $TSEED \
  --trainer.logger.init_args.name fp8recipe_${TAG}_$SIZE --trainer.logger.init_args.id fp8recipe-$TAG-$SIZE "--trainer.callbacks+=$KEEP"
