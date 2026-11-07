#!/bin/bash
#SBATCH -J gpic-mfu-gdecw
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:55:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GPU JPEG decoding a step ahead (GPIC_MFU_GPU_DECODE=2) at the wider widths, where the step should outrun the CPU decode but
# the shared bucket still waits on the DataLoader (w1866: d1024 8.2 ms of data wait per step and 13.9 ms per pull, d1280
# 8.8 and 14.2, straggler 4.3 and 12.6 ms): CPU decoding against GPU decoding in the ladder's setting (shared bucket, 128 x 1,
# Muon, A C D K K2; d1280 FSDP2 in groups of 7), full 8-GPU arms in turn (the CPUs are the resource under test).
# usage: gdec_wide.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/gdecw_$(date +%m%d_%H%M)
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
FS='{"class_path":"src.utils.fsdp_setup.NoSyncModelParallelStrategy","init_args":{"data_parallel_size":8,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}'
for W in "896 11 2 1536" "1024 12 3 1024"; do
  set -- $W
  MFU_MICRO=128 arm d$1_ddp_m128_cpudec $1 $2 $3 $4 "${MUON[@]}"
  GPIC_MFU_GPU_DECODE=2 MFU_MICRO=128 arm d$1_ddp_m128_gpuahead $1 $2 $3 $4 "${MUON[@]}"
done
GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=7 MFU_STRATEGY=$FS MFU_MICRO=128 arm d1280_fsdp7_m128_cpudec 1280 14 3 1024 "${MUON[@]}"
GPIC_MFU_GPU_DECODE=2 GPIC_MFU_FSDP=1 GPIC_MFU_FSDP_GROUP=7 MFU_STRATEGY=$FS MFU_MICRO=128 arm d1280_fsdp7_m128_gpuahead 1280 14 3 1024 "${MUON[@]}"
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
