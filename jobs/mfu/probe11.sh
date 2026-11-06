#!/bin/bash
#SBATCH -J gpic-mfu-probe11
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU per-width lever choice on the production path (docs/mfu.md; jobs/mfu/arm.sh, real Lite batches, shard seed 20261106,
# the same batches in every arm, the per-step loss of every rank): d640 levers off / A + C + D / A + C + D + E (bucketed
# packed captions, rebalanced) / DDP with FSDP2's per-block compile (the parity control for B); d1024 levers off / A /
# A + C + D / A + B + C + D. 1024 steps per arm.
# usage: probe11.sh <code worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/probe11_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PIXGEN_SRC=$WT/baselines/PixelGen
source $WT/jobs/mfu/gpu_checks.sh
gpu_checks $WT $OUT || { echo "GPU checks failed"; exit 3; }
export MFU_WT=$WT MFU_LIMIT=720 GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106
FSDP='{"class_path":"lightning.pytorch.strategies.ModelParallelStrategy","init_args":{"data_parallel_size":8,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}'
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; echo "[$(date +%T)] $1 rc=$?" | tee -a $OUT/summary.txt; }
arm d640_off 640 9 2 1024
GPIC_MFU_ATTN=fa3 GPIC_MFU_UINT8=1 GPIC_MFU_GPU_RESIZE=1 arm d640_ACD 640 9 2 1024
GPIC_MFU_ATTN=fa3 GPIC_MFU_UINT8=1 GPIC_MFU_GPU_RESIZE=1 GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=1 arm d640_ACDE 640 9 2 1024
GPIC_MFU_PERBLOCK_COMPILE=1 arm d640_perblock 640 9 2 1024
arm d1024_off 1024 12 3 1024
GPIC_MFU_ATTN=fa3 arm d1024_A 1024 12 3 1024
GPIC_MFU_ATTN=fa3 GPIC_MFU_UINT8=1 GPIC_MFU_GPU_RESIZE=1 arm d1024_ACD 1024 12 3 1024
GPIC_MFU_ATTN=fa3 GPIC_MFU_FSDP=1 MFU_STRATEGY=$FSDP GPIC_MFU_UINT8=1 GPIC_MFU_GPU_RESIZE=1 arm d1024_ABCD 1024 12 3 1024
cd $WT
python tools/mfu/bench_summary.py $OUT/d640_* $OUT/d1024_* --skip 100 | tee -a $OUT/summary.txt
python tools/mfu/parity_loss.py $OUT/d640_off $OUT/d640_perblock $OUT/d640_ACD $OUT/d640_ACDE | tee -a $OUT/summary.txt
python tools/mfu/parity_loss.py $OUT/d1024_off $OUT/d1024_A $OUT/d1024_ACD $OUT/d1024_ABCD | tee -a $OUT/summary.txt
echo "DONE $OUT"
