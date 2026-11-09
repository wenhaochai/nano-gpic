#!/bin/bash
#SBATCH -J gpic-fp8roundtrip
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Experiment B of the resume question (gpic-28 2026-11-09; owner popup 「A 和 B 都做」): the FP8 resume test's continuous arm again
# (fp8-d20-n3-s2c's recipe and seeds, one piece to 13,600), with an in-process checkpoint round trip at steps 200 and 11,904
# (src/callbacks/roundtrip.py: save through the trainer, wait for the write, load it back through the resume path; the process
# goes on). The early one fails fast if the mechanism breaks; the second is the test (resume arms 851.0 / 863.8 vs 808.3).
# usage: train_fp8_roundtrip.sh <tag> <fp8: none|native3> <shard seed> <torch seed>
export GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 4 ] || { echo "usage: train_fp8_roundtrip.sh <tag> <fp8: none|native3> <shard seed> <torch seed>"; exit 2; }
TAG=$1 F8=$2 SSEED=$3 TSEED=$4
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
[[ $TAG =~ ^[0-9a-z.-]+$ && $SSEED =~ ^[0-9]+$ && $TSEED =~ ^[0-9]+$ ]] || { echo "bad tag or seed"; exit 2; }
WD=1536 NB=16 NT=4 S=13600 MLR=6.8e-3 ALR=0.000179 WDK=0 GB=1024 CD=2720 WU=136 MB=128
SIZE=d$WD PAT='gpic_train_{00000..07999}.tar'
WT=$WC/project/gpic-wt/roundtrip_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
unset GPIC_INIT_CKPT GPIC_DEBUG_OPT_STATE
export GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=1 GPIC_MFU_MUON_ADD=1 GPIC_MFU_FP8=$F8
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_fp8recipe/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ ! -f $LADDER_RUN_ROOT/exp_ladder_$SIZE/last.ckpt ] || { echo "$LADDER_RUN_ROOT has a last.ckpt: this run must be in one piece"; exit 3; }
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SSEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
export BENCH_OUT=$LADDER_RUN_ROOT/bench
export GPIC_ROUNDTRIP_STEPS=200,11904
KEEP='{"class_path":"src.callbacks.roundtrip.RoundTrip"}'
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" --seed_everything $TSEED \
  --trainer.logger.init_args.name fp8recipe_${TAG}_$SIZE --trainer.logger.init_args.id fp8recipe-$TAG-$SIZE "--trainer.callbacks+=$KEEP"
