#!/bin/bash
#SBATCH -J gpic-mfu-gdecnarrow
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# gpu-decode-narrow (owner 2026-11-08 12:2x 「可以」 and popup 12:3x): lever C2 (GPU JPEG decoding a step ahead,
# GPIC_MFU_GPU_DECODE=2; d640 -18.3% per step on mfu, w1929/w1930; its FD arm passed) ported onto genmodel's sweep code
# (sweep-gpudec = sweep-noema-cap 8e1b9ef + mfu 301a3ae, 13568a5, 49b67af). The sweep's narrow shapes, where the CPU's JPEG
# decode binds (about 13.4k images/s on 96 cores, ~76 ms per 1024 images whatever the model): width 448 (21 + 2 blocks),
# 512 (16 + 2) and 640 (9 + 2), DDP, 128 x 1, the sweep's switches (NS, MA, no EMA, NVLS off). CPU decode and GPU decode
# as full 8-GPU arms in turn (the halves of a node would share its CPUs, the resource under test), the same batches,
# the order alternated per width. The decoding test on GPU 0 first.
# usage: gdec_narrow.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/mfu/gdec_narrow_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export PIXGEN_SRC=$WT/baselines/PixelGen MFU_WT=$WT GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106 MFU_LIMIT=1200
export GPIC_MFU_ATTN=fa3 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_UINT8=1 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_SHARED_BUCKET=1
export GPIC_MFU_NS_FUSED=1 GPIC_MFU_MUON_ADD=1 GPIC_MFU_NO_EMA=1 NCCL_NVLS_ENABLE=0
cd $WT
CUDA_VISIBLE_DEVICES=0 python tools/mfu/tests/test_gpu_decode.py 2>&1 | grep -v -i warn | tail -9 | tee -a $OUT/summary.txt
grep -q "GPU_DECODE_TEST PASS" $OUT/summary.txt || { echo "the GPU decoding test failed"; exit 1; }
MUON=(--config configs_h100/muon_ladder.yaml)
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; local rc=$?; echo "[$(date +%T)] $1 rc=$rc" | tee -a $OUT/summary.txt; }
cpu() { MFU_MICRO=128 arm d$1_cpudec $1 $2 $3 2048 "${MUON[@]}"; }
gpu() { GPIC_MFU_GPU_DECODE=2 MFU_MICRO=128 arm d$1_gpudec $1 $2 $3 2048 "${MUON[@]}"; }
gpu 448 21 2; cpu 448 21 2
cpu 512 16 2; gpu 512 16 2
gpu 640 9 2; cpu 640 9 2
python - $OUT <<'PY' | tee -a $OUT/summary.txt
import json, os, statistics as st, sys
out = sys.argv[1]
def rows(tag):
    f = os.path.join(out, tag, "bench", "steps_rank0.json")
    return json.load(open(f))["rows"] if os.path.exists(f) else None
ms = lambda x: x["step_ms"] if x.get("step_ms") is not None else 1000 * (x["t_end"] - x["t_start"])
for w in (448, 512, 640):
    a, b = rows(f"d{w}_cpudec"), rows(f"d{w}_gpudec")
    if not a or not b:
        print(f"width {w}: missing rows (CPU {bool(a)}, GPU {bool(b)})"); continue
    n = min(len(a), len(b)); ha, hb = a[n // 2:n], b[n // 2:n]
    same_T = sum(a[i].get("T") == b[i].get("T") for i in range(n))
    f = lambda r, k: st.mean(1000 * x.get(k, 0.0) for x in r)
    print(f"width {w}: mean step CPU {st.mean(map(ms, ha)):.1f} ms, GPU {st.mean(map(ms, hb)):.1f} ms "
          f"({st.mean(map(ms, hb)) / st.mean(map(ms, ha)) - 1:+.1%}); median {st.median(map(ms, ha)):.1f} vs {st.median(map(ms, hb)):.1f} "
          f"({st.median(map(ms, hb)) / st.median(map(ms, ha)) - 1:+.1%}); data wait {f(ha, 'data_wait_s'):.1f} vs {f(hb, 'data_wait_s'):.1f} ms; "
          f"loss over the last 200 steps {st.mean(x['loss'] for x in a[n - 200:n]):.5f} vs {st.mean(x['loss'] for x in b[n - 200:n]):.5f}; "
          f"the same bucket in {same_T} of {n} steps")
PY
grep -h "OutOfMemoryError\|CUDA out of memory\|Traceback" $OUT/*/train.log | sort | uniq -c | head -5
echo "DONE $OUT"
