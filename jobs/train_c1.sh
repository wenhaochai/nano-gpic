#!/bin/bash
#SBATCH -J gpic-resume-c1
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Resume isolation C1 (gpic-28 2026-11-09; owner popup 「R3 和 C1 都做」): the FP8 resume test's continuous arm again (its recipe and seeds, one
# piece to 13,600), but at micro-batch 11,904 the data path starts again inside the process as in a resumed launch (GPIC_DATA_RESTART_AT:
# the workers stop, the dataset builds a new loader whose shard stage skips every shard started so far, the bucket pools empty); nothing
# else restarts. A full-state copy at 8,500 (stable phase, before the cooldown at 10,880) is kept for a stable-phase resume test (genmodel's
# question on w2369/w2370). Then the official evaluation (eval_official_run9). If the data restart is the cause, FD reads like the resumes.
# usage: train_c1.sh <tag> <fp8: none|native3> <shard seed> <torch seed>
export GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 4 ] || { echo "usage: train_fp8_keep.sh <tag> <fp8: none|native3> <shard seed> <torch seed>"; exit 2; }
TAG=$1 F8=$2 SSEED=$3 TSEED=$4
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
[[ $TAG =~ ^[0-9a-z.-]+$ && $SSEED =~ ^[0-9]+$ && $TSEED =~ ^[0-9]+$ ]] || { echo "bad tag or seed"; exit 2; }
WD=1536 NB=16 NT=4 S=13600 MLR=6.8e-3 ALR=0.000179 WDK=0 GB=1024 CD=2720 WU=136 MB=128
SIZE=d$WD PAT='gpic_train_{00000..07999}.tar'
WT=$WC/project/gpic-wt/resume_isolate_run
export GPIC_DATA_RESTART_AT=11904
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
unset GPIC_INIT_CKPT GPIC_DEBUG_OPT_STATE
export GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=1 GPIC_MFU_MUON_ADD=1 GPIC_MFU_FP8=$F8
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_fp8recipe/$TAG LADDER_LAST_EVERY=$((S / 8))
mkdir -p $LADDER_RUN_ROOT/exp_ladder_$SIZE
[ ! -f $LADDER_RUN_ROOT/exp_ladder_$SIZE/last.ckpt ] || { echo "$LADDER_RUN_ROOT has a last.ckpt: this run must be in one piece"; exit 3; }
[ -s $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt ] || echo $SSEED > $LADDER_RUN_ROOT/exp_ladder_$SIZE/shard_seed.txt
export BENCH_OUT=$LADDER_RUN_ROOT/bench
KEEP='{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"last-keep-{step:06d}","monitor":null,"save_last":false,"save_top_k":-1,"every_n_train_steps":8500}}'
bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" --seed_everything $TSEED \
  --trainer.logger.init_args.name fp8recipe_${TAG}_$SIZE --trainer.logger.init_args.id fp8recipe-$TAG-$SIZE "--trainer.callbacks+=$KEEP"

rc=$?; echo "training rc=$rc"; [ $rc -eq 0 ] && [ -f $LADDER_RUN_ROOT/exp_ladder_$SIZE/ms-step=$(printf %06d $S).ckpt ] || exit 1
bash $WC/project/gpic-wt/eval_official_run9/jobs/evals_group.sh muon_fp8recipe/$TAG:$SIZE:$WD:$NB:$NT:$S
echo "eval rc=$?"
d=$WC/runs/gpic-speed/raw_view/muon_fp8recipe/$TAG/exp_ladder_$SIZE/eval_$(printf %06d $S)_euler_calls50_official
python3 -c "import json; m=json.load(open('$d/metrics.json')); i=json.load(open('$d/metrics_inception.json')); print('$TAG: FD %.2f FID %.2f precision %.4f recall %.4f' % (m['fd'], i['inception_fd'], m['precision'], m['recall']))"
echo RESUME_C1_DONE
