#!/bin/bash
#SBATCH -J gpic-mfu-ncclag
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:30:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# tools/mfu/nccl_ag_test.py under four NCCL settings: default, no NVLS, ring only, simple protocol only (gpic-28 2026-11-08,
# the d1536 final-layer fault: an out-of-place fp32 all-gather placed ranks' data at the wrong blocks). usage: nccl_ag_test.sh <tools worktree>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
cd ${1:?tools worktree}
source $WC/envs/pixgen/bin/activate
export TMPDIR=$WC/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p $TMPDIR; trap 'rm -rf $TMPDIR' EXIT
nvidia-smi topo -m 2>&1 | head -12
for e in "" "NCCL_NVLS_ENABLE=0" "NCCL_ALGO=Ring" "NCCL_PROTO=Simple"; do
  echo "== [$e]"
  env $e torchrun --standalone --nproc_per_node 8 tools/mfu/nccl_ag_test.py 2>&1 | grep -v -i warn | grep -E "NCCL |  n |DONE|Error|error" | head -34
done
