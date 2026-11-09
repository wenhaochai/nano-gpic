#!/bin/bash
#SBATCH -J gpic-resume-mech
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Resume isolation, the mechanism (gpic-28 2026-11-09). E1/E2 named the stream: with only the CPU streams reseeded after the warm-up
# the resume reads FD 867.0 (the resumed level, 851.0-884.1), with only the GPU streams reseeded 822.8 (seven one-piece runs:
# 802.7-830.0). Why a replayed GPU stream costs FD is open. This job resumes the kept full-state copy at 11,904 to 13,600 as E2 did
# (the warm-up, then torch.cuda.manual_seed_all(<GPU seed>) on every rank), with the seed as an argument:
#   <GPU seed> = the run's own torch seed (1): the stream of the first launch again, from its offset 0, so each real step draws what
#     the first launch drew 4 micro-batches earlier (R3's stream position, but with the warm-up run: R3 skipped it and read 827.0).
#     Clean: only the exact replay in step costs. At the resumed level: any replay of that stream costs and R3's result came from
#     the skipped warm-up.
#   <GPU seed> = 2, 3: two more fresh streams, the spread of the fixed resume (E2 is one run).
# GPIC_RNG_PROBE prints each stream's digest, seed and offset before and after the warm-up and after the reseed. Then the official
# evaluation (eval_official_run9, as E1/E2).
# usage: resume_mech.sh <continuous tag> <resumed tag> <fp8: none|native3> <GPU seed>
#   the planned calls: resume_mech.sh fp8-d20-n3-s2c fp8-d20-n3-s2rg1 native3 1 ; ... fp8-d20-n3-s2rg2 native3 2 ; ... fp8-d20-n3-s2rg3 native3 3
export GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 4 ] || { echo "usage: resume_mech.sh <continuous tag> <resumed tag> <fp8: none|native3> <GPU seed>"; exit 2; }
CT=$1 RT=$2 F8=$3 GSEED=$4
[[ $CT =~ ^[0-9a-z.-]+$ && $RT =~ ^[0-9a-z.-]+$ && $GSEED =~ ^[0-9]+$ ]] || { echo "bad tag or seed"; exit 2; }
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
WD=1536 NB=16 NT=4 S=13600 MLR=6.8e-3 ALR=0.000179 WDK=0 GB=1024 CD=2720 WU=136 MB=128
SIZE=d$WD PAT='gpic_train_{00000..07999}.tar'
WT=$WC/project/gpic-wt/resume_isolate_run4
export GPIC_RESEED_AFTER_WARMUP=$GSEED GPIC_RESEED_WHAT=cuda GPIC_RNG_PROBE=1
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
echo "resume test (GPU streams reseeded with $GSEED after the warm-up): $RT resumes $K (torch seed ${TSEED:-?}, fp8 [${F8:-none}])"
bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" --seed_everything ${TSEED:-1} \
  --trainer.logger.init_args.name fp8recipe_${RT}_$SIZE --trainer.logger.init_args.id fp8recipe-$RT-$SIZE

rc=$?; echo "training rc=$rc"; [ $rc -eq 0 ] && [ -f $LADDER_RUN_ROOT/exp_ladder_$SIZE/ms-step=$(printf %06d $S).ckpt ] || exit 1
bash $WC/project/gpic-wt/eval_official_run9/jobs/evals_group.sh muon_fp8recipe/$RT:$SIZE:$WD:$NB:$NT:$S
echo "eval rc=$?"
d=$WC/runs/gpic-speed/raw_view/muon_fp8recipe/$RT/exp_ladder_$SIZE/eval_$(printf %06d $S)_euler_calls50_official
python3 -c "import json; m=json.load(open('$d/metrics.json')); i=json.load(open('$d/metrics_inception.json')); print('$RT: FD %.2f FID %.2f precision %.4f recall %.4f' % (m['fd'], i['inception_fd'], m['precision'], m['recall']))" \
  && echo RESUME_MECH_DONE gpu-seed $GSEED
