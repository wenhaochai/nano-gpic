#!/bin/bash
#SBATCH -J gpic-mfu-v1
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU freeze check (d) for the full ladder (genmodel, 2026-11-06): loss parity over >= 1000 steps on the production path
# (jobs/mfu/arm.sh: Lightning, real GPIC Lite batches, the ladder's shard seed 20261106), the same batches in every arm
# (GPIC_MFU_DET_DATA=1). Arms: levers off (twice: is the reference run-to-run identical?), A alone (FA3; accepted at
# bf16 level in w1321: the floor), all levers (A + B FSDP2 + C + D + E), all but E. tools/mfu/parity_loss.py compares the per-step
# global loss (mean of the 8 ranks); step time and memory per arm from the step timer.
# usage: v1_parity.sh <code worktree> <d640 | d1280> <steps>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree} SZ=${2:?size} STEPS=${3:-1216}
case $SZ in d640) A=(640 9 2) ;; d1024) A=(1024 12 3) ;; d1280) A=(1280 14 3) ;; *) echo "size?"; exit 2 ;; esac
OUT=$WC/runs/gpic-speed/mfu/v1_${SZ}_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
IFS=, read -ra G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PIXGEN_SRC=$WT/baselines/PixelGen
( cd $WT && CUDA_VISIBLE_DEVICES=${G[0]} timeout 300 python tools/mfu/fa3_check.py > $OUT/fa3_check.log 2>&1 ) || { tail -20 $OUT/fa3_check.log; echo "FA3 check failed"; exit 3; }
export MFU_WT=$WT MFU_LIMIT=780 GPIC_MFU_DET_DATA=1 GPIC_SHARD_SEED=20261106
FSDP='{"class_path":"lightning.pytorch.strategies.ModelParallelStrategy","init_args":{"data_parallel_size":8,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}'
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; echo "[$(date +%T)] $1 rc=$?" | tee -a $OUT/summary.txt; }
arm off_a ${A[@]} $STEPS
GPIC_MFU_ATTN=fa3 arm A ${A[@]} $STEPS
GPIC_MFU_ATTN=fa3 GPIC_MFU_FSDP=1 MFU_STRATEGY=$FSDP GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=1 GPIC_MFU_UINT8=1 GPIC_MFU_GPU_RESIZE=1 arm all ${A[@]} $STEPS
GPIC_MFU_ATTN=fa3 GPIC_MFU_FSDP=1 MFU_STRATEGY=$FSDP GPIC_MFU_UINT8=1 GPIC_MFU_GPU_RESIZE=1 arm all_noE ${A[@]} $STEPS
[ $SZ = d640 ] && arm off_b ${A[@]} $STEPS
cd $WT
python tools/mfu/bench_summary.py $OUT/off_a $OUT/A $OUT/all $OUT/all_noE $( [ $SZ = d640 ] && echo $OUT/off_b ) --skip 100 | tee -a $OUT/summary.txt
python tools/mfu/parity_loss.py $OUT/off_a $OUT/A $OUT/all $OUT/all_noE $( [ $SZ = d640 ] && echo $OUT/off_b ) | tee -a $OUT/summary.txt
echo "DONE $OUT"
