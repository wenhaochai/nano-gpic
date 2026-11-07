#!/bin/bash
#SBATCH -J gpic-mfu-probe6
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU (docs/mfu.md), 8 GPUs on one node:
# 1. lever B correctness: 200 steps at d640 under torchrun with the same batches, t and noise, DDP vs FSDP2 (global mean
#    loss and grad norm per step), and DDP vs DDP (the run-to-run floor);
# 2. lever F: the production path at d1792 with last.ckpt every 100 steps (27 GB), synchronous Lightning saves vs the
#    async writer (GPIC_MFU_ASYNC_CKPT=1), 300 steps each;
# 3. levers A + B + E in step_bench at d1024 (FA3, FSDP2, packed + rebalanced), against DDP + SDPA.
# usage: probe6.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe6_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen
( cd $WT && CUDA_VISIBLE_DEVICES=${G[0]} timeout 300 python tools/mfu/fa3_check.py > $OUT/fa3_check.log 2>&1 ) || { tail -20 $OUT/fa3_check.log; echo "FA3 check failed"; exit 3; }
cd $WT
run() {   # run <tag> <step_bench args...>
  local tag=$1; shift
  timeout 600 torchrun --standalone --nproc_per_node 8 tools/mfu/step_bench.py --out $OUT/$tag.json "$@" > $OUT/$tag.log 2>&1
  echo "[$(date +%T)] $tag rc=$? $(grep -h MFU $OUT/$tag.log | tail -1)" | tee -a $OUT/summary.txt
}
run loss_ddp_a --d 640 --steps 200 --record
run loss_ddp_b --d 640 --steps 200 --record
run loss_fsdp --d 640 --steps 200 --record --fsdp
export MFU_WT=$WT MFU_LIMIT=780
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; local rc=$?; echo "[$(date +%T)] $1 rc=$rc" | tee -a $OUT/summary.txt; }
MFU_SAVE_EVERY=100 arm save_sync 1792 18 4 300
MFU_SAVE_EVERY=100 GPIC_MFU_ASYNC_CKPT=1 arm save_async 1792 18 4 300
python tools/mfu/bench_summary.py $OUT/save_sync $OUT/save_async --skip 20 | tee -a $OUT/summary.txt
for d in 1024; do
  run ddp_d${d}_base --d $d --steps 120
  GPIC_MFU_ATTN=fa3 run ddp_d${d}_ABE --d $d --steps 120 --fsdp --rebalance
done
echo "DONE $OUT"
