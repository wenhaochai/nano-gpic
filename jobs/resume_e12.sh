#!/bin/bash
#SBATCH -J gpic-resume-e12
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Resume isolation E1/E2 (gpic-28 2026-11-09; owner popup 「两个都做」): which restarted stream carries the resume cost? A plain resume
# restarts every stream at the first launch's state (the per-step loss noise replays the first launch's at lag 0, z 3.5); R3 replays the
# same GPU stream 4 micro-batches out of step and is clean (827); R4 reseeds everything and is clean (814). E1 (cpu): after the warm-up
# only torch's CPU generator (and numpy, random) is reseeded, so the data workers get new seeds while timesteps, noise and caption
# dropout replay in step. E2 (cuda): only the GPU generators are reseeded, so the worker seeds replay. The bad one (FD near 870) names
# the stream. GPIC_RNG_PROBE prints each stream's digest before and after the warm-up and the workers' base seed. Then the official
# evaluation (eval_official_run9).
# usage: resume_e12.sh <continuous tag> <resumed tag> <fp8: none|native3> <reseed: cpu|cuda>
export GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 4 ] || { echo "usage: resume_e12.sh <continuous tag> <resumed tag> <fp8: none|native3> <reseed: cpu|cuda>"; exit 2; }
CT=$1 RT=$2 F8=$3 WHAT=$4
case $WHAT in cpu|cuda) ;; *) echo "reseed must be cpu or cuda"; exit 2 ;; esac
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
WD=1536 NB=16 NT=4 S=13600 MLR=6.8e-3 ALR=0.000179 WDK=0 GB=1024 CD=2720 WU=136 MB=128
SIZE=d$WD PAT='gpic_train_{00000..07999}.tar'
WT=$WC/project/gpic-wt/resume_isolate_run3
export GPIC_RESEED_AFTER_WARMUP=987654321 GPIC_RESEED_WHAT=$WHAT GPIC_RNG_PROBE=1
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
echo "resume test ($WHAT reseed): $RT resumes $K (torch seed ${TSEED:-?}, fp8 [${F8:-none}])"
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
python3 -c "import json; m=json.load(open('$d/metrics.json')); i=json.load(open('$d/metrics_inception.json')); print('$RT: FD %.2f FID %.2f precision %.4f recall %.4f' % (m['fd'], i['inception_fd'], m['precision'], m['recall']))"
echo RESUME_E12_DONE $WHAT
