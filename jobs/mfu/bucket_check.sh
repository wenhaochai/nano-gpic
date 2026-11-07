#!/bin/bash
#SBATCH -J gpic-mfu-bucket
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:10:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The shared caption-length bucket on 8 GPUs (docs/mfu.md; GPIC_SHARED_BUCKET=1 = data.train_dataset shared_bucket):
# Muon overlay, accumulation 4, DDP with A C D K K2, 512 optimizer steps (2048 micro-batches) per arm:
# 1. step time without / with the bucket at d640, d1280, d1792;
# 2. d640 with the bucket to step 128 (last.ckpt), then a resume to 256;
# 3. every arm with the bucket: all 8 ranks' padded caption length T must agree micro-batch by micro-batch (the timer
#    logs T), also after the resume; 4. the bucket with micro-batch 128 x accumulation 1 at d640 and d1024.
# usage: bucket_check.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/bucket_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export PIXGEN_SRC=$WT/baselines/PixelGen MFU_WT=$WT GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106 MFU_LIMIT=1500
export GPIC_MFU_ATTN=fa3 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1
cd $WT
MUON=(--config configs_h100/muon_ladder.yaml)
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; echo "[$(date +%T)] $1 rc=$?" | tee -a $OUT/summary.txt; }
GPIC_SHARED_BUCKET=1 MFU_SAVE_EVERY=128 MFU_KEEP_CKPT=1 arm d640_S_to128 640 9 2 128 "${MUON[@]}"
CK=$(ls $OUT/d640_S_to128/exp_mfu_d640_S_to128/last*.ckpt $OUT/d640_S_to128/last*.ckpt 2>/dev/null | head -1)
echo "resume from: $CK" | tee -a $OUT/summary.txt
GPIC_SHARED_BUCKET=1 MFU_RESUME=$CK arm d640_S_resumed 640 9 2 256 "${MUON[@]}"
# the new ladders' combination: the bucket with micro-batch 128 x accumulation 1 (the same global batch 1024)
GPIC_SHARED_BUCKET=1 MFU_MICRO=128 arm d640_S_M128 640 9 2 512 "${MUON[@]}"
GPIC_SHARED_BUCKET=1 MFU_MICRO=128 arm d1024_S_M128 1024 12 3 512 "${MUON[@]}"
for d in "640 9 2" "1280 14 3" "1792 18 4"; do
  set -- $d
  arm d$1 $1 $2 $3 512 "${MUON[@]}"
  GPIC_SHARED_BUCKET=1 arm d$1_S $1 $2 $3 512 "${MUON[@]}"
done
python - $OUT <<'PY' | tee -a $OUT/summary.txt
import json, os, statistics as st, sys, glob
out = sys.argv[1]
for tag in ("d640_S_to128", "d640_S_resumed", "d640_S_M128", "d1024_S_M128", "d640", "d640_S", "d1280", "d1280_S", "d1792", "d1792_S"):
    fs = sorted(glob.glob(os.path.join(out, tag, "bench", "steps_rank*.json")))
    if not fs:
        print(f"{tag}: no timer rows"); continue
    rows = [json.load(open(f))["rows"] for f in fs]
    T = [[r.get("T") for r in rr] for rr in rows]
    n = min(len(t) for t in T)
    agree = sum(all(t[i] == T[0][i] for t in T) for i in range(n))
    ms = [x.get("step_ms", 1000 * (x["t_end"] - x["t_start"])) for x in rows[0]]
    half = ms[len(ms) // 2:]
    print(f"{tag}: {len(fs)} ranks, {n} micro-batches; all ranks the same T in {agree} of {n}; micro-batch "
          f"{st.mean(half):.2f} ms mean, {st.median(half):.2f} median (second half); mean T {st.mean([t for t in T[0][:n] if t]):.1f}")
PY
grep -h "Traceback\|Error" $OUT/*/train.log | sort | uniq -c | head -5
echo "DONE $OUT"
