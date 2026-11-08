#!/bin/bash
#SBATCH -J gpic-fp8recipe
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=03:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# Non-EMA hyper-parameter sweeps for the ladder playbook (owner 2026-11-07: no EMA, 「按方案排」; 21:5x: Muon for the matrices, AdamW for
# embeddings and scalars, 「按以前的分法」): a linear cooldown over the last <cooldown %> of the steps to an absolute 3e-5, 1% warmup, raw weights
# evaluated afterwards. Weight decay only on the Muon matrices. From scratch, resumable. Code: gpic-wt/muon_sweep_run (ladder-sweep c6fdbf6).
# Owner 2026-11-07 21:3x (all required): linear decay, an absolute LR floor of 3e-5, warmup = 1% of the run.
# gpic-28, the d20 FP8 check (owner popups 2026-11-08 03:4x 「做 d20 检验」 and 11:2x: the current recipe code, no EMA and the pool cap, both
# arms on the same code): genmodel's train_sweep_noema_cap.sh (tmp/ladder_prof/wsd, 2026-11-08 11:15) with one more argument, the FP8
# recipe: none (bf16) or native3 (src/models/layers/fp8_native.py, per-tensor e4m3, delayed scaling). Code: gpic-wt/sweep_fp8_cap_run
# (ladder-sweep-fp8-cap = sweep-noema-cap 8e1b9ef + FP8). Root runs/gpic-speed/muon_fp8recipe/<tag>.
# usage: train_sweep_fp8.sh <tag> <d> <nb> <nt> <steps> <muon lr> <adamw lr> <muon weight decay> <data: full|lite|s80|s8> <global batch> <cooldown %> <fp8: none|native3>
export GPIC_MFU_NO_EMA=1   # owner 2026-11-08 「ema是一定不能用」: no EMA update; checkpoints hold the raw weights in the EMA slot too
export NCCL_NVLS_ENABLE=0   # NCCL 2.26.2 NVLS all-gather can permute rank blocks (NVIDIA/nccl #1906; gpic-28 w2090)
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 12 ] || { echo "usage: train_sweep_fp8.sh <tag> <d> <nb> <nt> <steps> <muon lr> <adamw lr> <muon wd> <data> <global batch> <cooldown %> <fp8>"; exit 2; }
TAG=$1 WD=$2 NB=$3 NT=$4 S=$5 MLR=$6 ALR=$7 WDK=$8 DATA=$9 GB=${10} CDP=${11} F8=${12}
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
[[ $CDP =~ ^[0-9]+$ && $CDP -ge 5 && $CDP -le 90 ]] || { echo "bad cooldown %"; exit 2; }
[[ $GB =~ ^(256|512|1024|2048|4096)$ ]] || { echo "bad global batch"; exit 2; }
WU=$(( S / 100 )); [ $WU -ge 1 ] || WU=1
SIZE=d$WD
[[ $TAG =~ ^[0-9a-z.-]+$ && $S =~ ^[0-9]+$ && $((S % 32)) -eq 0 ]] || { echo "bad tag or steps"; exit 2; }
CD=$(( (S * CDP / 100) / 32 * 32 ))
case $DATA in full) PAT='gpic_train_{00000..07999}.tar' ;; lite) PAT='gpic_train_{00000..00800}.tar' ;;
  s80) PAT='gpic_train_{00000..00079}.tar' ;; s8) PAT='gpic_train_{00000..00007}.tar' ;; *) echo "bad data"; exit 2 ;; esac
SEED=20261106
WT=$WC/project/gpic-wt/sweep_fp8_cap_run
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
[ -f $WT/baselines/PixelGen/src/models/layers/fp8_native.py ] || { echo "$WT has no native FP8"; exit 2; }
# FSDP2 only for the ladder's wide sizes; every other shape is DDP at 128 images per GPU
case "$WD:$NB" in 1280:14) MB=128 FS=1 G=7 ;; 1024:24) MB=64 FS=0 G=1 ;; 1536:*) MB=128 FS=1 G=1 ;; 1792:*) MB=64 FS=1 G=1 ;; *) MB=128 FS=0 G=1 ;; esac
[ $((GB / 8)) -ge $MB ] || MB=$((GB / 8))   # small global batches: fewer images per GPU, no accumulation
export GPIC_MFU_FSDP=$FS GPIC_MFU_FSDP_GROUP=$G GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=$FS GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=$FS GPIC_MFU_MUON_ADD=1 GPIC_MFU_FP8=$F8
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_fp8recipe/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" \
  --trainer.logger.init_args.name fp8recipe_${TAG}_$SIZE --trainer.logger.init_args.id fp8recipe-$TAG-$SIZE
