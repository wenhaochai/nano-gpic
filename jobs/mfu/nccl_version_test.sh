#!/bin/bash
#SBATCH -J gpic-mfu-ncclver
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Which NCCL keeps NVLS on and the rank order right? (gpic-28 2026-11-08: NCCL 2.26.2's NVLS AllGather/ReduceScatter puts
# ranks' blocks out of order at 576 KB per rank, NVIDIA/nccl #1906 open, upstream fix PR #2377 closed unmerged; NVLS off costs
# ~2% at FSDP2 widths.) For each NCCL wheel unpacked in tmp/nccl_wheels/<version> (pip download nvidia-nccl-cu12), the plain
# all-gather and reduce-scatter tests under torch 2.7.1 with that libnccl preloaded, default settings (NVLS on); each test
# prints the NCCL version actually loaded. usage: nccl_version_test.sh <tools worktree> <code worktree>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
TOOLS=${1:?tools worktree} CODE=${2:?code worktree}
cd $TOOLS
source $WC/envs/pixgen/bin/activate
export TMPDIR=$WC/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p $TMPDIR; trap 'rm -rf $TMPDIR' EXIT
for d in $WC/tmp/nccl_wheels/*/; do
  lib=$d/nvidia/nccl/lib/libnccl.so.2
  [ -f $lib ] || continue
  echo "== $(basename $d)"
  LD_PRELOAD=$lib torchrun --standalone --nproc_per_node 8 tools/mfu/nccl_ag_test.py 2>&1 | grep -v -i warn | grep -E "NCCL \(|  n " | head -4
  LD_PRELOAD=$lib torchrun --standalone --nproc_per_node 8 tools/mfu/nccl_rs_test.py $CODE 2>&1 | grep -v -i warn | grep -E "bad cases|  n " | head -3
done
