#!/bin/bash
#SBATCH -J gpic-mfu-rebal
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Equal tokens per GPU for E at 128 x 1 (the gap breakdown, w1865: E's straggler wait at the gradient all-reduce is
# 16.1 ms per step at d640 and 6.6 at d1024, because random batches give the GPUs different real token counts). Lever
# E's deal of each batch position's samples over the GPUs (same math: every sample keeps its position, t and noise):
# R2 plans on the host one step ahead (src/data/rebalance_loader.py), R1 inside the step (src/utils/rebalance.py).
# Against the plain E + K2 arms of w1865 (the same batches before the deal).
# usage: rebal_check.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/rebal_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export PIXGEN_SRC=$WT/baselines/PixelGen MFU_WT=$WT GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106 MFU_LIMIT=1500
export GPIC_MFU_ATTN=fa3 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 BENCH_PROFILE_LIGHT=1
cd $WT
MUON=(--config configs_h100/muon_ladder.yaml)
FS='{"class_path":"src.utils.fsdp_setup.NoSyncModelParallelStrategy","init_args":{"data_parallel_size":8,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}'
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; local rc=$?; echo "[$(date +%T)] $1 rc=$rc" | tee -a $OUT/summary.txt; }
GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=2 GPIC_MFU_REBALANCE_SLOT_MB=512 MFU_MICRO=128 arm d640_ddp_m128_EKR2 640 9 2 2048 "${MUON[@]}" --data.train_dataset.init_args.group_window 128
GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=1 MFU_MICRO=128 arm d640_ddp_m128_EKR1 640 9 2 2048 "${MUON[@]}" --data.train_dataset.init_args.group_window 128
GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=2 GPIC_MFU_REBALANCE_SLOT_MB=512 MFU_MICRO=128 arm d1024_ddp_m128_EKR2 1024 12 3 1024 "${MUON[@]}" --data.train_dataset.init_args.group_window 128
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
    acc = 1024 // 8 // micro
    ms = [x.get("step_ms", 1000 * (x["t_end"] - x["t_start"])) for x in rows[0]][n // 2:]
    w = int(tag.split("_")[0][1:])
    opt_ms = acc * st.mean(ms)
    mem = max(max(r.get("max_reserved_gb", 0) for r in rr) for rr in rows)
    print(f"{tag}: micro {micro} x {acc}; optimizer step {opt_ms:.1f} ms (mean, second half), "
          f"MFU {GF[w] * 1e9 * 1024 / 8 / (opt_ms / 1e3) / 989.4e12:.1%}; peak reserved {mem:.1f} GiB; "
          f"{len(fs)} ranks the same T in {agree} of {n} micro-batches; mean T {st.mean([t for t in T[0][:n] if t] or [0]):.1f}")
PY
grep -h "OutOfMemoryError\|CUDA out of memory\|Traceback" $OUT/*/train.log | sort | uniq -c | head -5
echo "DONE $OUT"
