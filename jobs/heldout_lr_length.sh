#!/bin/bash
#SBATCH -J gpic-heldout-search
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:15:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss of the Muon lr x run length runs (stage 1 follow-up, 2026-11-07 04:4x; tmp/ladder_prof/submit_lr_length.sh): every ms
# checkpoint present under runs/gpic-speed/muon_lr_search2/mlr<tag>/exp_ladder_<size> when the job starts (d640 and d1024 to 1C0 at
# 1.7e-3 and 6.8e-3, d1280 to 22400 and d1792 to 8992 at 3.4e-3, 6.8e-3, 1.36e-2), EMA and raw, N 3072 + 3072, appended to
# tmp/heldout_loss/heldout_loss.jsonl (idempotent). The 3.4e-3 arms of d640 and d1024 are the Full ladder's runs (jobs/heldout_new_ladders.sh).
# Read run_root; train_flops assumes batch 256 (these runs use 1024).
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=($(ls $WC/runs/gpic-speed/muon_lr_search2/mlr*/exp_ladder_d*/ms-step=*.ckpt 2>/dev/null))
[ ${#CK[@]} -gt 0 ] || { echo "no ms checkpoints yet"; exit 0; }
echo "${#CK[@]} checkpoints"
HELDOUT_TRAIN_SHARDS="00787 00000" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
