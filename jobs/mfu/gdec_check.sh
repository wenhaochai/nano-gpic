#!/bin/bash
#SBATCH -J gpic-mfu-gdec
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GPU JPEG decoding (GPIC_MFU_GPU_DECODE=1; owner 2026-11-07: 可以用GPU解码) at the widths the CPU's JPEG decoding binds
# (d640 and d768: about 13.4k images/s on 96 cores whatever the model, median step 61 ms against a mean of 76 ms at d640):
# the ladder's own setting (shared bucket, 128 x 1, Muon, A C D K K2): CPU decoding, GPU decoding in the step
# (GPIC_MFU_GPU_DECODE=1: the training thread waits for nvJPEG, about 8 ms a step) and GPU decoding a step ahead in a
# background thread (=2), full 8-GPU arms in turn (the two halves of a node would share its CPUs, the resource under test).
# The decoding test on GPU 0 first.
# usage: gdec_check.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/gdec_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export PIXGEN_SRC=$WT/baselines/PixelGen MFU_WT=$WT GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106 MFU_LIMIT=1500
export GPIC_MFU_ATTN=fa3 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_SHARED_BUCKET=1
cd $WT
CUDA_VISIBLE_DEVICES=0 python tools/mfu/tests/test_gpu_decode.py 2>&1 | grep -v -i warn | tail -7 | tee -a $OUT/summary.txt
grep -q "GPU_DECODE_TEST PASS" $OUT/summary.txt || { echo "the GPU decoding test failed"; exit 1; }
MUON=(--config configs_h100/muon_ladder.yaml)
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; local rc=$?; echo "[$(date +%T)] $1 rc=$rc" | tee -a $OUT/summary.txt; }
for W in "640 9 2" "768 10 2"; do
  set -- $W
  MFU_MICRO=128 arm d$1_ddp_m128_cpudec $1 $2 $3 2048 "${MUON[@]}"
  GPIC_MFU_GPU_DECODE=1 MFU_MICRO=128 arm d$1_ddp_m128_gpudec $1 $2 $3 2048 "${MUON[@]}"
  GPIC_MFU_GPU_DECODE=2 MFU_MICRO=128 arm d$1_ddp_m128_gpuahead $1 $2 $3 2048 "${MUON[@]}"
done
python - $OUT <<'PY' | tee -a $OUT/summary.txt
import json, os, statistics as st, sys, glob
out = sys.argv[1]
GF = {640: 104.1, 768: 163.8, 896: 242.1, 1024: 346.9, 1280: 622.4, 1536: 1024.9, 1792: 1554.8}
for d in sorted(glob.glob(os.path.join(out, "d*")), key=lambda p: (int(os.path.basename(p).split("_")[0][1:]), p)):
    tag = os.path.basename(d)
    fs = sorted(glob.glob(os.path.join(d, "bench", "steps_rank*.json")))
    if not fs:
        print(f"{tag}: no timer rows (failed: see train.log)"); continue
    rows = [json.load(open(f))["rows"] for f in fs]
    T = [[r.get("T") for r in rr] for rr in rows]
    n = min(len(t) for t in T)
    agree = sum(all(t[i] == T[0][i] for t in T) for i in range(n))
    micro = 128 if "m128" in tag else 64
    rows0 = rows[0]; n0 = len(rows0)
    med = st.median(x.get("step_ms", 1000 * (x["t_end"] - x["t_start"])) for x in rows0[n0 // 2:])
    dw = 1000 * st.mean(x.get("data_wait_s", 0.0) for x in rows0[n0 // 2:])
    acc = 1024 // 8 // micro
    ms = [x.get("step_ms", 1000 * (x["t_end"] - x["t_start"])) for x in rows[0]][n // 2:]
    w = int(tag.split("_")[0][1:])
    opt_ms = acc * st.mean(ms)
    mem = max(max(r.get("max_reserved_gb", 0) for r in rr) for rr in rows)
    print(f"{tag}: micro {micro} x {acc}; optimizer step {opt_ms:.1f} ms (mean, second half), "
          f"MFU {GF[w] * 1e9 * 1024 / 8 / (opt_ms / 1e3) / 989.4e12:.1%}; peak reserved {mem:.1f} GiB; "
          f"median step {med:.1f} ms; data wait {dw:.1f} ms (rank 0, mean); {len(fs)} ranks the same T in {agree} of {n} micro-batches")
PY
grep -h "OutOfMemoryError\|CUDA out of memory\|Traceback" $OUT/*/train.log | sort | uniq -c | head -5
echo "DONE $OUT"
