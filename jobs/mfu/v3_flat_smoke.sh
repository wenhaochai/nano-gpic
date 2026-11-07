#!/bin/bash
#SBATCH -J gpic-mfu-v3flat
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# MFU lever B3's checkpoint path (docs/mfu.md B3): jobs/mfu/v2c_resume.sh with V2_MODES=flat: the full-data-path smoke
# (d640, stop at 1000, resume to 4800, tools/check_shard_resume.py) with flat FSDP2 units + B2; the ms and last
# checkpoints' layout against 5cd3372's smoke (the same keys and shapes); gmflow_run4's evaluation on ms-step=001600.
# usage: v3_flat_smoke.sh <code worktree>
export V2_MODES=flat
exec bash "${1:?code worktree}/jobs/mfu/v2c_resume.sh" "$@"   # wq runs a copy of this script: find the others in the worktree
