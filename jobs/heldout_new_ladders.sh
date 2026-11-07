#!/bin/bash
#SBATCH -J gpic-heldout-search
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss of the new ladders (owner 2026-11-07: the repetition arms are evaluated with Euler 50 + held-out loss): every ms
# checkpoint present under runs/gpic-speed/muon_{full,lite,s80,s8}/exp_ladder_<size> when the job starts, EMA and raw, N 3072 + 3072,
# appended to tmp/heldout_loss/heldout_loss.jsonl (idempotent: a rerun computes only the checkpoints that lack lines). Read run_root:
# the 'ladder' field says lite for these roots, and train_flops assumes batch 256 (these runs use 1024). The train split is
# gpic_train_00787, which s80 and s8 never read (for them it is a second held-out set).
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=($(ls $WC/runs/gpic-speed/muon_{full,lite,s80,s8}/exp_ladder_d*/ms-step=*.ckpt 2>/dev/null))
[ ${#CK[@]} -gt 0 ] || { echo "no ms checkpoints yet"; exit 0; }
echo "${#CK[@]} checkpoints"
heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
