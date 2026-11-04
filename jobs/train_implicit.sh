#!/bin/bash
#SBATCH -J gpic-implicit
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# One run of the implicit sampler on the ladder's recipe (docs/distribution_step.md): jobs/train_ladder.sh with the overlay
# configs_h100/implicit_train.yaml. d640 at C0/4: fork before block 6 of 9, 2 samples per x_t, so 137.8 GFLOP per image against the point
# model's 104.2 (tools/ladder_flops.py's formula), and 133600 x 104.2 / 137.8 = 101031, so 101024 steps (a multiple of 32) for the same training FLOPs.
# usage: train_implicit.sh [main.py arguments...]    Run: exp_ladder_implicit_d640. Code: the frozen worktree gpic-wt/dist_step_run.
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
export LADDER_WT=${LADDER_WT:-$WC/project/gpic-wt/dist_step_run}
bash $LADDER_WT/jobs/train_ladder.sh ${IMPLICIT_NAME:-implicit_d640} 640 9 2 ${IMPLICIT_STEPS:-101024} --config configs_h100/implicit_train.yaml "$@"
