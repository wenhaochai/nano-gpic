#!/bin/bash
#SBATCH -J gpic-mfu-probe8
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU levers B and F with checkpoints (docs/mfu.md), production path at d640, 8 GPUs:
# 1. FSDP2 (ModelParallelStrategy) + async writes, last.ckpt every 100 steps, 200 steps; then a resume from that
#    last.ckpt to step 300; tools/mfu/check_ckpt.py: the file loads, the eval keys (denoiser.*, ema_denoiser.*) have the
#    production shapes, the resumed run starts at step 200, the EMA in the file equals the EMA the run held;
# 2. the same with DDP and Lightning's synchronous saves (control).
# usage: probe8.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe8_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
export PIXGEN_SRC=$WT/baselines/PixelGen MFU_WT=$WT MFU_LIMIT=600 MFU_KEEP_CKPT=1 MFU_SAVE_EVERY=100
FSDP='{"class_path":"lightning.pytorch.strategies.ModelParallelStrategy","init_args":{"data_parallel_size":8,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}'
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; echo "[$(date +%T)] $1 rc=$?" | tee -a $OUT/summary.txt; }
last() { ls -t $OUT/$1/last*.ckpt $OUT/$1/exp_mfu_$1/last*.ckpt 2>/dev/null | head -1; }
GPIC_MFU_FSDP=1 MFU_STRATEGY=$FSDP GPIC_MFU_ASYNC_CKPT=1 arm fsdp_a 640 9 2 200
GPIC_MFU_FSDP=1 MFU_STRATEGY=$FSDP GPIC_MFU_ASYNC_CKPT=1 MFU_RESUME=$(last fsdp_a) arm fsdp_b 640 9 2 300
arm ddp_a 640 9 2 200
MFU_RESUME=$(last ddp_a) arm ddp_b 640 9 2 300
cd $WT/baselines/PixelGen && PYTHONPATH=$PWD python $WT/tools/mfu/check_ckpt.py $(last fsdp_a) $(last ddp_a) $OUT 2>&1 | tee -a $OUT/summary.txt
rm -f $OUT/*/last*.ckpt $OUT/*/*/last*.ckpt $OUT/*/*.ckpt $OUT/*/*/*.ckpt
echo "DONE $OUT"
