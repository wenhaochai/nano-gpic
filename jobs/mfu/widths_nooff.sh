#!/bin/bash
#SBATCH -J gpic-mfu-widths
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# jobs/mfu/widths.sh without the levers-off arm of each width (WIDTHS_NO_OFF=1: the first set is the reference): fewer
# arms, less idle GPU time between them. usage: widths_nooff.sh <code worktree> <steps> <width>:<set>[,<set>...] ...
export WIDTHS_NO_OFF=1
exec bash "$(dirname "$(readlink -f "$0")")/widths.sh" "$@"
