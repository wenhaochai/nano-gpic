#!/bin/bash
# The final ladder of the post "How to Scale Text-to-image Models: a Handbook" (wenhaochai.com/blogs/scaling-generative-models-101.html):
# one run per compute budget, at the size the fitted scaling law picks (about 50 image tokens per image parameter), with the post's
# recipe: Muon for the matrices, AdamW for the rest; Muon lr 1.36e-2 x (C / 3.6e18)^-0.5; AdamW lr 6.4e-4 x (parameters / 109.4M)^-0.547;
# Muon weight decay 5 / (Muon lr x steps), AdamW 0; batch 1024 images; warmup over the first 1% of the steps and a linear decay to 3e-5
# over the last 50%; no EMA (the raw weights are scored); GPIC Full, one seeded shard order per pass.
# jobs/final_ladder_sizes.sh holds the values of the post's seven runs (W&B reself/gpic-speed); this script passes them to
# jobs/train_ladder.sh with the speed settings those runs used (FSDP2 from width 1280, fused norms, FlashAttention 3).
# One 8-GPU node per run. Started again with the same arguments, a run resumes from its newest last*.ckpt.
#
# usage: GPIC_WC=<work root> bash jobs/final_ladder.sh <d10|d11|d12|d13|d14|d16|d18> [main.py arguments...]
#   <work root>/data/gpic/train/      GPIC Full's 8,000 training shards gpic_train_00000.tar ... gpic_train_07999.tar
#   <work root>/data/gpic/val/        the validation shards gpic_val_*.tar
#   <work root>/data/gpic/train_emb_qwen3_1.7b/   the captions' Qwen3-1.7B features (jobs/encode_captions.sh)
#   outputs: <work root>/runs/final_ladder/<size>/ (GPIC_OUT moves it); checkpoints ms-step=<step>.ckpt at 1/4, 1/2, 3/4 and the end.
#   GPIC_DATA moves the data directory; GPIC_VENV picks the Python environment (empty: the python on PATH).
#   W&B logs offline unless WANDB_MODE says otherwise.
#   Extra arguments go to main.py after the recipe's, e.g. --trainer.max_steps 320 for a short test (the schedule keeps the full
#   run's length; set LADDER_PARTIAL_OK=1 so that a launch stopped early counts as done).
set -eo pipefail
SIZE=${1:?usage: final_ladder.sh <d10|d11|d12|d13|d14|d16|d18> [main.py arguments...]}; shift
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
WC=${GPIC_WC:?set GPIC_WC to the work root (see the usage above)}
DATA=${GPIC_DATA:-$WC/data/gpic}
source $HERE/jobs/final_ladder_sizes.sh
final_ladder_size $SIZE || { echo "unknown size $SIZE: $FINAL_LADDER_SIZES"; exit 2; }
for d in train val train_emb_qwen3_1.7b; do [ -d "$DATA/$d" ] || { echo "missing $DATA/$d"; exit 2; }; done
WU=$((S / 100)) CD=$((S / 2 / 32 * 32)) ACC=$((1024 / 8 / MB))
# FSDP2 from width 1280 (width 1280 shards in groups of 7 blocks), DDP below it, as the post's runs
case $D in 1280) FS=1 G=7 ;; 1536|1792) FS=1 G=1 ;; *) FS=0 G=1 ;; esac
export GPIC_WC=$WC GPIC_REPO=$HERE GPIC_ENV_SH=$HERE/jobs/env.sh
export GPIC_MFU_NO_EMA=1   # no EMA update: the checkpoints' EMA slot holds the raw weights
export GPIC_MFU_FSDP=$FS GPIC_MFU_FSDP_GROUP=$G GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_LOCAL_OPT=$FS GPIC_MFU_PATCHIFY=$FS \
       GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_MUON_ADD=1
export NCCL_NVLS_ENABLE=0   # NCCL 2.26's NVLS all-gather can permute rank blocks (NVIDIA/nccl #1906)
export WANDB_MODE=${WANDB_MODE:-offline}
export LADDER_WT=$HERE LADDER_RUN_ROOT=${GPIC_OUT:-$WC/runs/final_ladder}/$SIZE LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_d$D
[ -s $LADDER_RUN_ROOT/exp_ladder_d$D/shard_seed.txt ] || echo 20261106 > $LADDER_RUN_ROOT/exp_ladder_d$D/shard_seed.txt
exec bash $HERE/jobs/train_ladder.sh d$D $D $NB $NT $S \
  --config configs_h100/muon_ladder.yaml \
  --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD \
  --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $ACC \
  --data.train_dataset.init_args.shared_bucket true --data.train_dataset.init_args.tar_pattern "gpic_train_{00000..07999}.tar" \
  --data.train_dataset.init_args.tar_dir $DATA/train/ --data.train_dataset.init_args.emb_dir $DATA/train_emb_qwen3_1.7b \
  --model.conditioner.init_args.emb_dir $DATA/train_emb_qwen3_1.7b --data.val_dataset.init_args.tar_dir $DATA/val/ \
  --torch_hub_dir $WC/.cache/torch/hub --huggingface_cache_dir $WC/cache/huggingface/hub \
  --trainer.logger.init_args.name final_ladder_$SIZE --trainer.logger.init_args.id final-ladder-$SIZE \
  "$@"
