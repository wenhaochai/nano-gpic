#!/bin/bash
#SBATCH -J gpic-mfu-fsdpperm
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Which side permutes final_layer.linear's row blocks at d1536 (tools/mfu/fsdp_perm_test.py): the training's FSDP2 setup on
# 8 GPUs, the unsharded weight inside one forward against full_tensor() and the checkpoint. d1536 27200 (G 1) without and
# with compiled blocks, d1280 22400 (G 7) as the control. Code: the ladder worktree given (run2 0da7c56: the d1536 C0 code).
# usage: fsdp_perm_test.sh <code worktree> <tools worktree>
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
CODE=${1:?code worktree} TOOLS=${2:?tools worktree}
export TMPDIR=$WC/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p $TMPDIR; trap 'rm -rf $TMPDIR' EXIT
export TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
source $WC/envs/pixgen/bin/activate
export GPIC_MFU_ATTN=fa3 GPIC_MFU_FUSED_NORM=1 GPIC_MFU_FUSED_QK=1 GPIC_MFU_PATCHIFY=1 GPIC_MFU_FSDP=1
R=$WC/runs/gpic-speed/muon_full
cd $TOOLS
for spec in "1 $R/exp_ladder_d1536/ms-step=027200.ckpt" "1 $R/exp_ladder_d1536/ms-step=027200.ckpt compile" "7 $R/exp_ladder_d1280/ms-step=022400.ckpt"; do
  set -- $spec
  GPIC_MFU_FSDP_GROUP=$1 torchrun --standalone --nproc_per_node 8 tools/mfu/fsdp_perm_test.py $CODE $2 $3 2>&1 | grep -v -i -E "warn|^\s*$" | tail -14
done
