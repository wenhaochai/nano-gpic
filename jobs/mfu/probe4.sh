#!/bin/bash
#SBATCH -J gpic-mfu-probe4
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever E (docs/mfu.md): packed captions and per-position rebalancing over the ranks, on FA3 (lever A).
# 1. single-GPU tasks from a queue: whole-model parity of the packed path (tools/mfu/test_attn_parity.py --lever packed);
#    GPU-only step time, FA3 padded vs FA3 packed, at four widths;
# 2. 8 GPUs under torchrun, each rank its own T: FA3 padded (control), packed, packed + rebalanced, FA3 padded with one T
#    for all ranks (the no-imbalance bound), FA3 padded again; at d640 and d1280;
# 3. loss parity under DDP, 200 steps at d640: the global mean loss of FA3 padded vs packed + rebalanced (same batches,
#    t and noise; a sample keeps its batch position).
# usage: probe4.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe4_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[ ${#G[@]} -eq 8 ] || { echo "need 8 GPUs, have ${CUDA_VISIBLE_DEVICES:-}"; exit 2; }
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen GPIC_MFU_ATTN=fa3
source $WT/jobs/mfu/pool.sh
cd $WT
CUDA_VISIBLE_DEVICES=${G[0]} timeout 300 python tools/mfu/fa3_check.py > $OUT/fa3_check.log 2>&1 || { tail -20 $OUT/fa3_check.log; echo "FA3 check failed"; exit 3; }
SB="timeout 900 python $WT/tools/mfu/step_bench.py"
{
  for d in 1792 1280 1024 640; do
    echo "$SB --d $d --steps 80 --out $OUT/s_d${d}_pad.json > $OUT/s_d${d}_pad.log 2>&1; $SB --d $d --steps 80 --packed --out $OUT/s_d${d}_packed.json > $OUT/s_d${d}_packed.log 2>&1"
  done
  for args in "--d 640" "--d 640 --compile" "--d 1280 --nb 2 --nt 1" "--d 1792 --nb 2 --nt 1 --T 168"; do
    echo "timeout 600 python $WT/tools/mfu/test_attn_parity.py --lever packed $args > $OUT/parity_$(echo $args | tr -d ' -').log 2>&1"
  done
} > $OUT/tasks.txt
pool $OUT/tasks.txt $OUT
grep -h "PARITY" $OUT/parity_*.log
run() {   # run <tag> <step_bench args...>
  local tag=$1; shift
  timeout 600 torchrun --standalone --nproc_per_node 8 tools/mfu/step_bench.py --out $OUT/$tag.json "$@" > $OUT/$tag.log 2>&1
  echo "[$(date +%T)] $tag rc=$? $(grep -h MFU $OUT/$tag.log | tail -1)" | tee -a $OUT/summary.txt
}
for d in 640 1280; do
  run ddp_d${d}_pad1 --d $d --steps 150
  run ddp_d${d}_packed --d $d --steps 150 --packed
  run ddp_d${d}_rebal --d $d --steps 150 --rebalance
  run ddp_d${d}_same --d $d --steps 150 --same-T
  run ddp_d${d}_pad2 --d $d --steps 150
done
run loss_d640_pad --d 640 --steps 200 --record
run loss_d640_rebal --d 640 --steps 200 --record --rebalance
echo "DONE $OUT"
