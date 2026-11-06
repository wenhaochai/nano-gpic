#!/bin/bash
#SBATCH -J gpic-mfu-probe1
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU work (docs/mfu.md): 1. attention backends at the ladder's shapes (tools/mfu/attn_bench.py) under torch 2.7.1 (no
# FA3) and torch 2.15 (envs/pixgen-next: cuDNN 9.26, FlexAttention); 2. the GPU-only training step (tools/mfu/step_bench.py,
# synthetic batches, no data path, one GPU each) at six widths, side by side; 3. the same step on 8 GPUs under DDP at d640
# and d1280, with each rank's own T and with one T for all ranks (the cost of waiting for the rank with the longest
# captions). Measures only.
# usage: probe1.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe1_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[ ${#G[@]} -eq 8 ] || { echo "need 8 GPUs, have ${CUDA_VISIBLE_DEVICES:-}"; exit 2; }
export TORCHINDUCTOR_CPP_WRAPPER=1 PIXGEN_SRC=$WT/baselines/PixelGen
cd $WT
# 1 + 2: eight single-GPU processes
CUDA_VISIBLE_DEVICES=${G[0]} timeout 1500 python tools/mfu/attn_bench.py --backends sdpa,sdpa_cudnn,flex --out $OUT/attn_t27.json > $OUT/attn_t27.log 2>&1 &
( source $WC/envs/pixgen-next/bin/activate && export TORCHINDUCTOR_CACHE_DIR=$WC/cache/inductor/gpic_t215 TRITON_CACHE_DIR=$WC/cache/triton/gpic_t215 && \
  CUDA_VISIBLE_DEVICES=${G[1]} timeout 1500 python tools/mfu/attn_bench.py --backends sdpa,sdpa_cudnn,flex --out $OUT/attn_t215.json ) > $OUT/attn_t215.log 2>&1 &
i=2
for d in 640 896 1024 1280 1536 1792; do
  CUDA_VISIBLE_DEVICES=${G[$i]} timeout 1500 python tools/mfu/step_bench.py --d $d --steps 80 --out $OUT/step1_d$d.json > $OUT/step1_d$d.log 2>&1 &
  i=$((i + 1))
done
wait
echo "[$(date +%T)] phase 1 done"
# 3: DDP on 8 GPUs
for d in 640 1280; do
  for mode in own same; do
    extra=""; [ $mode = same ] && extra=--same-T
    timeout 900 torchrun --standalone --nproc_per_node 8 tools/mfu/step_bench.py --d $d --steps 120 $extra --out $OUT/ddp_d${d}_$mode.json > $OUT/ddp_d${d}_$mode.log 2>&1
    echo "[$(date +%T)] ddp d$d $mode rc=$?"
  done
done
grep -h "MFU\|Error\|error" $OUT/*.log | head -40
echo "DONE $OUT"
