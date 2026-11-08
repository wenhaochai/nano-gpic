#!/bin/bash
#SBATCH -J gpic-fp8branch-heldout
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss of every ms checkpoint (848, 1696, 2544, 3392) of the FP8 branch runs (gpic-28; w2311's held-out pass had no WT and wrote
# nothing): 20 tasks on 8 GPUs, the cooldown's course in each branch. usage: fp8_branch_heldout.sh <tag>...
set -eo pipefail
shopt -s nullglob
WC=/scratch/gpfs/GROUP/USER
export NCCL_NVLS_ENABLE=0
source $WC/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/heldout_loss_run
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=(); for TAG in "$@"; do CK+=($WC/runs/gpic-speed/muon_fp8branch/$TAG/exp_ladder_d1536/ms-step=*.ckpt); done
echo "${#CK[@]} checkpoints"
HELDOUT_TRAIN_SHARDS="00787" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
