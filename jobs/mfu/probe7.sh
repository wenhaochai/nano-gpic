#!/bin/bash
#SBATCH -J gpic-mfu-probe7
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever G0 (docs/mfu.md): Inductor's autotuning of the pointwise and reduction kernels (no GEMM autotuning), on the
# FA3 step (lever A), GPU-only step time on one GPU each, A / A + max_autotune_pointwise / A + that + coordinate-descent
# tuning / A again, at four widths; single-GPU tasks from a queue (jobs/mfu/pool.sh). Same math (other block sizes change
# the order of fp32 reductions only).
# usage: probe7.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe7_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen GPIC_MFU_ATTN=fa3
source $WT/jobs/mfu/pool.sh
cd $WT
CUDA_VISIBLE_DEVICES=${G[0]} timeout 300 python tools/mfu/fa3_check.py > $OUT/fa3_check.log 2>&1 || { tail -20 $OUT/fa3_check.log; echo "FA3 check failed"; exit 3; }
SB="timeout 1500 python $WT/tools/mfu/step_bench.py"
{
  # one task per width and variant: plain / variant / plain on the same GPU (A/B/A); separate Inductor caches, so an
  # autotuned kernel is never served to the plain arm or the reverse
  for d in 1792 1280 1024 640; do
    for v in pw cd; do
      C=$WC/cache/inductor/mfu_g0_d${d}_$v
      venv="TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1"
      [ $v = cd ] && venv="$venv TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1"
      echo "TORCHINDUCTOR_CACHE_DIR=${C}_plain $SB --d $d --steps 80 --out $OUT/d${d}_${v}_a1.json > $OUT/d${d}_${v}_a1.log 2>&1; TORCHINDUCTOR_CACHE_DIR=${C}_tuned $venv $SB --d $d --steps 80 --out $OUT/d${d}_${v}.json > $OUT/d${d}_${v}.log 2>&1; TORCHINDUCTOR_CACHE_DIR=${C}_plain $SB --d $d --steps 80 --out $OUT/d${d}_${v}_a2.json > $OUT/d${d}_${v}_a2.log 2>&1"
    done
  done
} > $OUT/tasks.txt
pool $OUT/tasks.txt $OUT
grep -h "MFU" $OUT/*.log
echo "DONE $OUT"
