#!/bin/bash
#SBATCH -J gpic-mfu-ncclrs
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:20:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# nccl-rs-check (owner popup 2026-11-08 09:3x): tools/mfu/nccl_rs_test.py (reduce-scatter and all-reduce correctness at the
# ladder's FSDP2 unit sizes and a sweep) under the default NCCL settings and with NCCL_NVLS_ENABLE=0; then the all-gather
# test again for the record. usage: nccl_rs_test.sh <tools worktree> <code worktree>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
TOOLS=${1:?tools worktree} CODE=${2:?code worktree}
cd $TOOLS
source $WC/envs/pixgen/bin/activate
export TMPDIR=$WC/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p $TMPDIR; trap 'rm -rf $TMPDIR' EXIT
for e in "" "NCCL_NVLS_ENABLE=0"; do
  echo "== reduce-scatter / all-reduce [$e]"
  env $e torchrun --standalone --nproc_per_node 8 tools/mfu/nccl_rs_test.py $CODE 2>&1 | grep -v -i warn | grep -E "NCCL |bad cases|  n |DONE|Error|error" | head -45
done
echo "== all-gather [default]"
torchrun --standalone --nproc_per_node 8 tools/mfu/nccl_ag_test.py 2>&1 | grep -v -i warn | grep -E "NCCL |DONE" | head -3
