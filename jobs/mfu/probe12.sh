#!/bin/bash
#SBATCH -J gpic-mfu-probe12
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:20:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever H (docs/mfu.md): the clip multiply, AdamW and the EMA in one Triton kernel (src/utils/fused_adamw_ema.py).
# 1. One GPU per task: tools/mfu/tests/test_fused_adamw_ema.py (bit for bit against torch, every FMA placement; time at
#    d1024's and d1792's parameter shapes); step_bench plain / H1 / H2 / plain at d1280, d1024, d640 on the same GPU;
#    end to end bit for bit at d640 with SDPA's math backend (deterministic): plain twice, H2 once, 30 recorded steps.
# 2. Production path (jobs/mfu/widths.sh, 8 GPUs, the same batches): d1024 off / ACD / ACDH1 / ACDH2, d640 off / ACD /
#    ACDH2, 1024 steps.
# usage: probe12.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe12_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen GPIC_MFU_ATTN=fa3
source $WT/jobs/mfu/pool.sh
cd $WT
SB="timeout 1200 python $WT/tools/mfu/step_bench.py"
{
  for d in 1280 1024 640; do
    echo "$SB --d $d --steps 80 --out $OUT/sb_d${d}_a1.json > $OUT/sb_d${d}_a1.log 2>&1; $SB --d $d --steps 80 --fused-opt 1 --out $OUT/sb_d${d}_h1.json > $OUT/sb_d${d}_h1.log 2>&1; $SB --d $d --steps 80 --fused-opt 2 --out $OUT/sb_d${d}_h2.json > $OUT/sb_d${d}_h2.log 2>&1; $SB --d $d --steps 80 --out $OUT/sb_d${d}_a2.json > $OUT/sb_d${d}_a2.log 2>&1"
  done
  R="GPIC_MFU_ATTN=sdpa GPIC_MFU_SDPA_BACKEND=MATH $SB --d 640 --steps 30 --warm 0 --record"
  echo "$R --out $OUT/rec_plain1.json > $OUT/rec_plain1.log 2>&1"
  echo "$R --out $OUT/rec_plain2.json > $OUT/rec_plain2.log 2>&1"
  echo "$R --fused-opt 2 --out $OUT/rec_h2.json > $OUT/rec_h2.log 2>&1"
  echo "python tools/mfu/tests/test_fused_adamw_ema.py --d 1024 > $OUT/test_d1024.log 2>&1"
  echo "python tools/mfu/tests/test_fused_adamw_ema.py --d 1792 > $OUT/test_d1792.log 2>&1"
} > $OUT/tasks.txt
pool $OUT/tasks.txt $OUT
grep -h "MFU\|FUSED_OPT_TEST\|bitwise variant\| ms per step" $OUT/*.log
python - $OUT <<'PY'
import json, sys, os
o = sys.argv[1]
r = {k: json.load(open(f"{o}/rec_{k}.json")) for k in ("plain1", "plain2", "h2") if os.path.exists(f"{o}/rec_{k}.json")}
if len(r) == 3:
    for k in ("plain2", "h2"):
        same = [r[k][f] == r["plain1"][f] for f in ("loss", "grad_norm", "param_sum", "param_abs", "ema_sum")]
        print(f"end to end d640 (SDPA math backend) {k} == plain1: loss, grad_norm, param_sum, param_abs, ema_sum {same}")
for d in (1280, 1024, 640):
    v = {k: json.load(open(f"{o}/sb_d{d}_{k}.json"))["mean_ms"] for k in ("a1", "h1", "h2", "a2") if os.path.exists(f"{o}/sb_d{d}_{k}.json")}
    print(f"d{d} one GPU mean step ms: {v}")
PY
bash $WT/jobs/mfu/widths.sh $WT 1024 1024:ACD,ACDH1,ACDH2 640:ACD,ACDH2
echo "DONE $OUT"
