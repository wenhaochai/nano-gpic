#!/bin/bash
#SBATCH -J gpic-mfu-probe10
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever E after the fix (docs/mfu.md): the packed caption rows bucketed to multiples of 128 (cuBLASLt heuristics ran
# before every launch with a new M, w1382). 1. one GPU each, all at once: padded vs packed at d640, d1024, d1280, d1792
# (FA3 in both); 2. 8 GPUs under torchrun, each rank its own T: padded vs packed + rebalanced at d640 and d1280.
# usage: probe10.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe10_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen GPIC_MFU_ATTN=fa3
cd $WT
SB="timeout 1200 python $WT/tools/mfu/step_bench.py --steps 80"
i=0
for d in 640 1024 1280 1792; do
  for m in pad packed; do
    extra=""; [ $m = packed ] && extra=--packed
    ( CUDA_VISIBLE_DEVICES=${G[$i]} $SB --d $d $extra --out $OUT/s_d${d}_$m.json > $OUT/s_d${d}_$m.log 2>&1; echo "[$(date +%T)] s_d${d}_$m rc=$?" >> $OUT/summary.txt ) &
    i=$((i + 1))
  done
done
wait
run() {   # run <tag> <step_bench args...>
  local tag=$1; shift
  timeout 600 torchrun --standalone --nproc_per_node 8 tools/mfu/step_bench.py --out $OUT/$tag.json "$@" > $OUT/$tag.log 2>&1
  echo "[$(date +%T)] $tag rc=$? $(grep -h MFU $OUT/$tag.log | tail -1)" | tee -a $OUT/summary.txt
}
for d in 640 1280; do
  run ddp_d${d}_pad --d $d --steps 150
  run ddp_d${d}_rebal --d $d --steps 150 --rebalance
done
grep -h MFU $OUT/s_*.log
echo "DONE $OUT"
