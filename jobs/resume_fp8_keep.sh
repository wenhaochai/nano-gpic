#!/bin/bash
#SBATCH -J gpic-fp8keep-resume
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The paired arm of the FP8 resume test (owner popup 2026-11-08 23:4x): a fresh root gets the continuous arm's full-state copy at 11,904 as
# its last.ckpt, with its shard seed and shard logs (train_ladder.sh drops the shards started after the checkpoint's shard clock), and
# resumes it to 13,600 with the continuous arm's settings: the same state at 11,904, then a resume (a new process, a restarted data
# stream, a fresh FP8 delayed-scaling history, another RNG position) against no resume. Per-step losses in <root>/bench.
# usage: resume_fp8_keep.sh <continuous tag> <resumed tag> <fp8: none|native3>
export GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 3 ] || { echo "usage: resume_fp8_keep.sh <continuous tag> <resumed tag> <fp8: none|native3>"; exit 2; }
CT=$1 RT=$2 F8=$3
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
WD=1536 NB=16 NT=4 S=13600 MLR=6.8e-3 ALR=0.000179 WDK=0 GB=1024 CD=2720 WU=136 MB=128
SIZE=d$WD PAT='gpic_train_{00000..07999}.tar'
WT=$WC/project/gpic-wt/fp8_branch_run
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SRC=$WC/runs/gpic-speed/muon_fp8recipe/$CT/exp_ladder_$SIZE
K=$(ls $SRC/last-keep-step=*.ckpt 2>/dev/null | head -1); [ -n "$K" ] || { echo "no full-state copy in $SRC"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/muon_fp8recipe/$RT LADDER_LAST_EVERY=$((S / 8))
DST=$LADDER_RUN_ROOT/exp_ladder_$SIZE
mkdir -p $DST
[ -f $DST/last.ckpt ] || { cp "$K" $DST/last.ckpt && cp $SRC/shard_seed.txt $SRC/shards_started_rank*.txt $DST/; } || exit 2
cp -n $SRC/config-*.yaml $DST/ 2>/dev/null || true
unset GPIC_INIT_CKPT GPIC_DEBUG_OPT_STATE
export GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=1 GPIC_MFU_FSDP_NOSYNC=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=1 GPIC_MFU_NS_FUSED=1 GPIC_MFU_LOCAL_OPT=1 GPIC_MFU_MUON_ADD=1 GPIC_MFU_FP8=$F8
export BENCH_OUT=$LADDER_RUN_ROOT/bench
TSEED=$(grep -h -o -E "seed_everything: [0-9]+" $SRC/config-*.yaml 2>/dev/null | head -1 | grep -o "[0-9]*$")
echo "resume test: $RT resumes $K (torch seed ${TSEED:-?}, fp8 [${F8:-none}])"
exec bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" --seed_everything ${TSEED:-1} \
  --trainer.logger.init_args.name fp8recipe_${RT}_$SIZE --trainer.logger.init_args.id fp8recipe-$RT-$SIZE
