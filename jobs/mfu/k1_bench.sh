#!/bin/bash
#SBATCH -J gpic-mfu-k1
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU levers K and K2 (docs/mfu.md; src/models/layers/fused_norm.py, fused_qk.py): the image blocks' residual add +
# RMSNorm + adaLN modulation (K) and their attention inputs, q/k RMSNorm + RoPE + the k/v concatenation (K2), as Triton
# kernels, forward and backward. One GPU per task (jobs/mfu/pool.sh):
# 1. the training step on one GPU (tools/mfu/step_bench.py, compiled, C++ launcher, FA3 on) at d1024-d1792: A, A + K and
#    A + K + K2, the same seed and batches, per-step loss recorded (the early-step difference is the rounding onset);
# 2. tools/mfu/tests/test_fused_norm.py and test_fused_qk.py on the GPU (forward correctly rounded up to the approximate
#    rsqrt, gradients at bf16 rounding level and below the eager ops' error, deterministic, the model's sizes);
# 3. tools/mfu/bench_fused_norm.py: each site alone against Inductor's compiled ops, and the backward's launch sweep.
# usage: k1_bench.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/k1_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PIXGEN_SRC=$WT/baselines/PixelGen TORCHINDUCTOR_CPP_WRAPPER=1 GPIC_MFU_ATTN=fa3
source $WT/jobs/mfu/pool.sh
cd $WT
S="timeout 1200 python tools/mfu/step_bench.py --steps 100 --record"
for d in 1792 1536 1280 1024; do
  echo "$S --d $d --out $OUT/d${d}_A.json > $OUT/d${d}_A.log 2>&1"
  echo "GPIC_MFU_FUSED_NORM=1 $S --d $d --out $OUT/d${d}_AK.json > $OUT/d${d}_AK.log 2>&1"
  echo "GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 $S --d $d --out $OUT/d${d}_AKK2.json > $OUT/d${d}_AKK2.log 2>&1"
done > $OUT/tasks.txt
echo "timeout 600 python tools/mfu/tests/test_fused_norm.py > $OUT/test.log 2>&1" >> $OUT/tasks.txt
echo "timeout 600 python tools/mfu/tests/test_fused_qk.py > $OUT/test_qk.log 2>&1" >> $OUT/tasks.txt
echo "timeout 900 python tools/mfu/bench_fused_norm.py > $OUT/kbench.log 2>&1" >> $OUT/tasks.txt
pool $OUT/tasks.txt $OUT
for d in 1792 1536 1280 1024; do
  python - $OUT/d${d}_A.json $OUT/d${d}_AK.json $OUT/d${d}_AKK2.json <<'PY'
import json, os, sys
runs = [json.load(open(f)) if os.path.exists(f) else {} for f in sys.argv[1:4]]
la = runs[0].get("loss", [])
out = [os.path.basename(sys.argv[1])[:-7]]
for name, r in zip(("A", "A+K", "A+K+K2"), runs):
    lk = r.get("loss", [])
    diff = [abs(x - y) / max(abs(x), 1e-12) for x, y in zip(la, lk)]
    pick = lambda i: f"{diff[i]:.1e}" if i < len(diff) else "-"
    out.append(f"{name} {r.get('mean_ms', 0):.2f} / {r.get('median_ms', 0):.2f} ms" + ("" if name == "A" else
               " (loss rel diff to A at steps 0, 1, 2, 10, 40, 99: " + ", ".join(pick(i) for i in (0, 1, 2, 10, 40, 99)) + ")"))
print("; ".join(out))
PY
done | tee $OUT/summary.txt
tail -2 $OUT/test.log; tail -2 $OUT/test_qk.log; grep "saves\|best" $OUT/kbench.log
echo "DONE $OUT"
