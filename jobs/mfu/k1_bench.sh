#!/bin/bash
#SBATCH -J gpic-mfu-k1
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:35:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever K (docs/mfu.md; src/models/layers/fused_norm.py): the residual add, RMSNorm and adaLN modulation of the image
# blocks as one Triton kernel per site, forward and backward. One GPU per task (jobs/mfu/pool.sh):
# 1. the training step on one GPU (tools/mfu/step_bench.py, compiled, C++ launcher, FA3 on) at d1024-d1792, A against
#    A + K, the same seed and batches, per-step loss recorded (the early-step difference is the rounding onset);
# 2. tools/mfu/tests/test_fused_norm.py on the GPU (forward correctly rounded, gradients at bf16 rounding level and below
#    the eager ops' error, deterministic, up to C = 1792);
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
done > $OUT/tasks.txt
echo "timeout 600 python tools/mfu/tests/test_fused_norm.py > $OUT/test.log 2>&1" >> $OUT/tasks.txt
echo "timeout 900 python tools/mfu/bench_fused_norm.py > $OUT/kbench.log 2>&1" >> $OUT/tasks.txt
pool $OUT/tasks.txt $OUT
for d in 1792 1536 1280 1024; do
  python - $OUT/d${d}_A.json $OUT/d${d}_AK.json <<'PY'
import json, sys
a, k = (json.load(open(f)) for f in sys.argv[1:3])
la, lk = a.get("loss", []), k.get("loss", [])
diff = [abs(x - y) / max(abs(x), 1e-12) for x, y in zip(la, lk)]
pick = lambda i: f"{diff[i]:.1e}" if i < len(diff) else "-"
print(f"{sys.argv[1].split('/')[-1][:-7]}: A {a.get('mean_ms', 0):.2f} / {a.get('median_ms', 0):.2f} ms, A+K {k.get('mean_ms', 0):.2f} / "
      f"{k.get('median_ms', 0):.2f} ms (mean / median); loss rel diff at steps 0, 1, 2, 10, 40, 99: "
      + ", ".join(pick(i) for i in (0, 1, 2, 10, 40, 99)))
PY
done | tee $OUT/summary.txt
tail -3 $OUT/test.log; grep "saves\|best" $OUT/kbench.log
echo "DONE $OUT"
