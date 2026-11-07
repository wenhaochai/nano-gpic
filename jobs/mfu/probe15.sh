#!/bin/bash
#SBATCH -J gpic-mfu-probe15
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# What the per-rank caption length costs now (docs/mfu.md): each rank pads its 32 captions to its own longest, the step
# waits for the slowest rank. w1316 (levers off, DDP) measured +5.9 ms at d640 and +6.1 ms at d1280 for "own T per rank"
# against "one T for all ranks". Here with today's levers (A, K, K2; FSDP2 at d1024-d1792, DDP at d640), tools/mfu/
# step_bench.py under torchrun, synthetic batches from the measured T histogram, 200 steps: own T against --same-T.
# usage: probe15.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe15_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen GPIC_MFU_ATTN=fa3 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1
cd $WT
run() {   # run <tag> <step_bench args...>
  local tag=$1; shift
  timeout 900 torchrun --standalone --nproc_per_node 8 tools/mfu/step_bench.py --steps 200 --out $OUT/$tag.json "$@" > $OUT/$tag.log 2>&1
  echo "[$(date +%T)] $tag rc=$? $(grep -h MFU $OUT/$tag.log | tail -1)" | tee -a $OUT/summary.txt
}
for d in 1024 1280 1792; do
  run fsdp_d${d}_ownT --d $d --fsdp
  run fsdp_d${d}_sameT --d $d --fsdp --same-T
done
run ddp_d640_ownT --d 640
run ddp_d640_sameT --d 640 --same-T
echo "DONE $OUT"
