#!/bin/bash
#SBATCH -J gpic-mfu-probe3
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever B (docs/mfu.md): the communication and optimizer part of the step, on 8 GPUs under torchrun with synthetic
# batches (tools/mfu/step_bench.py), each rank its own T as in training. Arms, each against controls on the same node:
# DDP as production; DDP with an AVG all-reduce comm hook (no per-parameter divide kernels); DDP with 250 MiB buckets;
# FSDP2 (per-block fully_shard, bf16 compute params, fp32 reduce-scatter, fused AdamW and EMA on the shards).
# usage: probe3.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe3_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen
cd $WT
run() {   # run <tag> <step_bench args...>
  local tag=$1; shift
  timeout 600 torchrun --standalone --nproc_per_node 8 tools/mfu/step_bench.py --steps 150 --out $OUT/$tag.json "$@" > $OUT/$tag.log 2>&1
  echo "[$(date +%T)] $tag rc=$? $(grep -h MFU $OUT/$tag.log | tail -1)" | tee -a $OUT/summary.txt
}
run d1280_base1 --d 1280
run d1280_avg --d 1280 --hook avg
run d1280_fsdp --d 1280 --fsdp
run d1280_b250 --d 1280 --bucket-mb 250
run d1280_base2 --d 1280
run d1792_base1 --d 1792
run d1792_avg --d 1792 --hook avg
run d1792_fsdp --d 1792 --fsdp
run d1792_base2 --d 1792
echo "DONE $OUT"
