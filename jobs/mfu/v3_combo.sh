#!/bin/bash
#SBATCH -J gpic-mfu-v3combo
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:20:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Two MFU checks in one host slot (docs/mfu.md): 1. lever B3's checkpoint path (jobs/mfu/v3_flat_smoke.sh: stop/resume
# with flat FSDP2 units + B2, checkpoint layout, evaluation path); 2. lever H1 with the aligned kernel at the DDP widths
# (jobs/mfu/widths_nooff.sh, 2048 steps: d768 and d640, A+C+D against A+C+D+H1).
# usage: v3_combo.sh <code worktree>
WT=${1:?code worktree}
D=$WT/jobs/mfu   # wq runs a copy of this script: find the others in the worktree
V2_MODES=flat bash $D/v2c_resume.sh $WT
WIDTHS_NO_OFF=1 bash $D/widths.sh $WT 2048 768:ACD,ACDH1 640:ACD,ACDH1
