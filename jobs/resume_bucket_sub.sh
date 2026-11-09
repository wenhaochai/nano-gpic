#!/bin/bash
#SBATCH -J gpic-bucketsub-resume
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Causal test of the resume artifact (gpic-28 2026-11-09; owner popup 09:0x 「修复并测试」): the FP8 resume test's resumed arm (fp8-d20-n3-s2r:
# FD 851.0 against 808.3 continuous; held-out equal) trained its last 1696 steps without the longest caption bucket, because every new
# process starts with empty bucket queues (shared_bucket_loader.py). This arm repeats that resume (the continuous arm's full-state copy at
# 11,904, its shard seed and shard logs, its settings) with GPIC_SHARED_BUCKET_SUB=8 (sub-batches of 16; in the loader test bucket 320
# first comes at micro-batch 279 instead of 1950), then the official evaluation of the raw weights at 13,600 with the code that scored
# both arms (eval_official_run5). Pass: FD within 1% of 808.3. Per-step T in <root>/bench. No held-out: one checkpoint would idle 7 GPUs.
# usage: resume_bucket_sub.sh <continuous tag> <resumed tag> <fp8: none|native3>
export GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
[ $# -eq 3 ] || { echo "usage: resume_fp8_keep.sh <continuous tag> <resumed tag> <fp8: none|native3>"; exit 2; }
CT=$1 RT=$2 F8=$3
case $F8 in none) F8= ;; native3) ;; *) echo "fp8 must be none or native3"; exit 2 ;; esac
WD=1536 NB=16 NT=4 S=13600 MLR=6.8e-3 ALR=0.000179 WDK=0 GB=1024 CD=2720 WU=136 MB=128
SIZE=d$WD PAT='gpic_train_{00000..07999}.tar'
WT=$WC/project/gpic-wt/bucket_sub_run
export GPIC_SHARED_BUCKET_SUB=8
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
echo "bucket-sub resume test: $RT resumes $K (torch seed ${TSEED:-?}, fp8 [${F8:-none}], sub ${GPIC_SHARED_BUCKET_SUB})"
bash $WT/jobs/train_ladder.sh $SIZE $WD $NB $NT $S \
  --config configs_h100/muon_ladder.yaml --model.optimizer.init_args.lr $ALR --model.optimizer.init_args.muon_lr $MLR \
  --model.optimizer.init_args.weight_decay 0 --model.optimizer.init_args.muon_weight_decay $WDK \
  --model.lr_scheduler.init_args.num_warmup_steps $WU --model.lr_scheduler.init_args.num_cooldown_steps $CD --model.lr_scheduler.init_args.max_steps $S --model.lr_scheduler.init_args.min_lr 3e-5 \
  --data.train_batch_size $MB --trainer.accumulate_grad_batches $((GB / 8 / MB)) --data.train_dataset.init_args.shared_bucket true \
  --data.train_dataset.init_args.tar_pattern "$PAT" --seed_everything ${TSEED:-1} \
  --trainer.logger.init_args.name fp8recipe_${RT}_$SIZE --trainer.logger.init_args.id fp8recipe-$RT-$SIZE
rc=$?; echo "training rc=$rc"; [ $rc -eq 0 ] && [ -f $DST/ms-step=$(printf %06d $S).ckpt ] || exit 1
source $WC/project/gpic/jobs/env.sh
RAW_SRC=muon_wsd/../muon_fp8recipe/$RT python -I $WC/tmp/ladder_prof/make_raw_view.py $SIZE $S || exit 1
bash $WC/project/gpic-wt/eval_official_run5/jobs/evals_gmflow.sh 0 $SIZE $WD $NB $NT $WC/runs/gpic-speed/raw_view/muon_fp8recipe/$RT \
  --wt eval_official_run5 $S:euler:50
echo "eval rc=$?"
python - $WC/runs/gpic-speed $CT $RT <<'PY'
import json, os, sys
R, ct, rt = sys.argv[1:4]
for tag in (ct, ct[:-1] + "r", rt):
    d = f"{R}/raw_view/muon_fp8recipe/{tag}/exp_ladder_d1536/eval_013600_euler_calls50_official"
    m = json.load(open(f"{d}/metrics.json")) if os.path.exists(f"{d}/metrics.json") else {}
    i = json.load(open(f"{d}/metrics_inception.json")) if os.path.exists(f"{d}/metrics_inception.json") else {}
    print(f"{tag}: FD {m.get('fd', float('nan')):.2f} FID {i.get('inception_fd', float('nan')):.2f} precision {m.get('precision', float('nan')):.4f} recall {m.get('recall', float('nan')):.4f}")
T = {}
for r in range(8):
    for x in json.load(open(f"{R}/muon_fp8recipe/{rt}/bench/steps_rank{r}.json"))["rows"]:
        T.setdefault(x["step"], []).append(x["T"])
steps = sorted(T)
for e in (144, 176, 320):
    first = next((s for s in steps if e in T[s]), None)
    share = 100 * sum(t == e for s in steps for t in T[s]) / (8 * len(steps))
    print(f"{rt}: bucket {e} first at step {first} ({None if first is None else first - steps[0]} after the resume), share {share:.2f}%")
PY
echo BUCKET_SUB_RESUME_DONE
