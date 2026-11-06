#!/bin/bash
#SBATCH -J gpic-mfu-prof1
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:55:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU work, step 1 (docs/mfu.md): where the GPU time of a production ladder step goes at four widths, on one node.
# Each arm: 200 steps of the production path (jobs/mfu/arm.sh), step timer with per-stage CUDA events, torch.profiler over
# steps 150-154 on all ranks. d1280 also logs NCCL's init and tuning tables. Measures only; no checkpoints, no W&B.
# usage: prof1.sh <code worktree at a clean commit>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?code worktree}
OUT=$WC/runs/gpic-speed/mfu/prof1_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
nvidia-smi -q -d CLOCK,PERFORMANCE > $OUT/nvidia_smi_start.txt 2>&1
export MFU_WT=$WT MFU_LIMIT=900 BENCH_STAGES=1 BENCH_PROFILE=150:155 TORCH_LOGS=recompiles
arm() { bash $WT/jobs/mfu/arm.sh $OUT "$@"; echo "[$(date +%T)] $1 rc=$?"; }
arm d640 640 9 2 200
arm d1024 1024 12 3 200
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING arm d1280 1280 14 3 200
arm d1792 1792 18 4 200
nvidia-smi -q -d CLOCK,PERFORMANCE > $OUT/nvidia_smi_end.txt 2>&1
echo "DONE $OUT"
