#!/bin/bash
#SBATCH -J gpic-mfu-probe5
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:55:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU levers on the production path (docs/mfu.md): Lightning, real GPIC Lite batches, the production settings of
# jobs/train_ladder.sh (jobs/mfu/arm.sh), 8 GPUs, 300 steps per arm, no checkpoints. Arms per width, on one node:
# production; A (FA3); A + E (packed captions, rebalanced over the ranks); A + D + E (and uint8 crops); production again.
# usage: probe5.sh <code worktree> <width tag> (d640 | d1280)
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree} SZ=${2:?size}
case $SZ in d640) A=(640 9 2) ;; d1024) A=(1024 12 3) ;; d1280) A=(1280 14 3) ;; d1792) A=(1792 18 4) ;; *) echo "size?"; exit 2 ;; esac
OUT=$WC/runs/gpic-speed/mfu/probe5_${SZ}_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PIXGEN_SRC=$WT/baselines/PixelGen
( cd $WT && CUDA_VISIBLE_DEVICES=${G[0]} timeout 300 python tools/mfu/fa3_check.py > $OUT/fa3_check.log 2>&1 ) || { tail -20 $OUT/fa3_check.log; echo "FA3 check failed"; exit 3; }
export MFU_WT=$WT MFU_LIMIT=780
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; echo "[$(date +%T)] $1 rc=$?" | tee -a $OUT/phases.txt; }
arm base1 ${A[@]} 300
GPIC_MFU_ATTN=fa3 arm A ${A[@]} 300
GPIC_MFU_ATTN=fa3 GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=1 arm AE ${A[@]} 300
GPIC_MFU_ATTN=fa3 GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=1 GPIC_MFU_UINT8=1 arm ADE ${A[@]} 300
arm base2 ${A[@]} 300
python $WT/tools/mfu/bench_summary.py $OUT/base1 $OUT/A $OUT/AE $OUT/ADE $OUT/base2 --skip 50 | tee $OUT/summary.txt
echo "DONE $OUT"
