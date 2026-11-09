#!/bin/bash
#SBATCH -J gpic-rankseed-d16
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Per-rank random streams at width 1536 (gpic-28 2026-11-09). Lightning seeds every rank alike, so at each step the 8 ranks draw the
# same timesteps, noise and caption dropout: a global batch of 1024 holds 128 different (t, noise) draws, each on 8 images.
# GPIC_RANK_SEED=1 (src/callbacks/host_tuning.py: torch.manual_seed per rank after the warm-up; the data workers start before it and
# keep their seeds) gives 1024 different draws. The one test before (width 640 to C0/8, 2026-11-03, one pair, no noise estimate)
# read FD +2.2% and -3.5%. Here the reference has a spread: seven one-piece runs of this recipe with these seeds read FD
# 802.7-830.0 (mean 814.2). This job is one more of them with the switch on, in one piece (a resume would replay the streams), then
# the official evaluation (eval_official_run9, as the reference). The training code is the reference's (gpic-wt/fp8_branch_run).
# usage: train_rankseed.sh <tag> <fp8: none|native3> <shard seed> <torch seed>
#   the planned calls: train_rankseed.sh fp8-d20-n3-rk1 native3 20261109 1 ; ... fp8-d20-n3-rk2 ... ; ... fp8-d20-n3-rk3 ...
# --time: 13,600 steps at 0.31 s (70 min) + the start (~4 min) + the evaluation (~12 min).
export GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0 GPIC_RANK_SEED=1
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 4 ] || { echo "usage: train_rankseed.sh <tag> <fp8: none|native3> <shard seed> <torch seed>"; exit 2; }
TAG=$1 F8=$2 SSEED=$3 TSEED=$4
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
[[ $TAG =~ ^[0-9a-z.-]+$ && $SSEED =~ ^[0-9]+$ && $TSEED =~ ^[0-9]+$ ]] || { echo "bad tag or seed"; exit 2; }
WD=1536 NB=16 NT=4 S=13600 MLR=6.8e-3 ALR=0.000179 WDK=0 GB=1024 CD=2720 WU=136 MB=128
SIZE=d$WD PAT='gpic_train_{00000..07999}.tar'
WT=$WC/project/gpic-wt/fp8_branch_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q GPIC_RANK_SEED $WT/baselines/PixelGen/src/callbacks/host_tuning.py || { echo "$WT has no GPIC_RANK_SEED"; exit 2; }
unset GPIC_INIT_CKPT GPIC_DEBUG_OPT_STATE
export GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=1 GPIC_MFU_MUON_ADD=1 GPIC_MFU_FP8=$F8
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_fp8recipe/$TAG LADDER_LAST_EVERY=$((S / 8))
RUN=$LADDER_RUN_ROOT/exp_ladder_$SIZE
mkdir -p $RUN
[ ! -f $RUN/last.ckpt ] || { echo "$LADDER_RUN_ROOT has a last.ckpt: this run must be in one piece"; exit 3; }
[ -s $RUN/shard_seed.txt ] || echo $SSEED > $RUN/shard_seed.txt
export BENCH_OUT=$LADDER_RUN_ROOT/bench
bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" --seed_everything $TSEED \
  --trainer.logger.init_args.name fp8recipe_${TAG}_$SIZE --trainer.logger.init_args.id fp8recipe-$TAG-$SIZE

rc=$?; echo "training rc=$rc"; [ $rc -eq 0 ] && [ -f $RUN/ms-step=$(printf %06d $S).ckpt ] || exit 1
n=$(grep -c " size=" $RUN/launches.txt); [ "$n" = 1 ] || { echo "$n launches in $RUN/launches.txt: not one piece"; exit 1; }
bash $WC/project/gpic-wt/eval_official_run9/jobs/evals_group.sh muon_fp8recipe/$TAG:$SIZE:$WD:$NB:$NT:$S
echo "eval rc=$?"
d=$WC/runs/gpic-speed/raw_view/muon_fp8recipe/$TAG/exp_ladder_$SIZE/eval_$(printf %06d $S)_euler_calls50_official
python3 -c "import json; m=json.load(open('$d/metrics.json')); i=json.load(open('$d/metrics_inception.json')); print('$TAG: FD %.2f FID %.2f precision %.4f recall %.4f' % (m['fd'], i['inception_fd'], m['precision'], m['recall']))" \
  && echo RANKSEED_DONE $TAG
