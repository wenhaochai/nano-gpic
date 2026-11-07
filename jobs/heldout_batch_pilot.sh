#!/bin/bash
#SBATCH -J gpic-heldout-search
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:15:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss of the batch pilot and the seed repeats (owner 2026-11-07 popup: 「先做 batch 试验」, 「跑两次」): every ms checkpoint under
# runs/gpic-speed/muon_batch_pilot/<tag>/exp_ladder_<size> and runs/gpic-speed/muon_bucket_ab/<tag>/exp_ladder_d640, EMA and raw, N 3072 + 3072,
# appended to tmp/heldout_loss/heldout_loss.jsonl (idempotent). Code: heldout_loss_run @ 7449b49.
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss_run   # frozen at 7449b49
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=($(ls $WC/runs/gpic-speed/muon_batch_pilot/*/exp_ladder_d*/ms-step=*.ckpt $WC/runs/gpic-speed/muon_bucket_ab/*/exp_ladder_d640/ms-step=*.ckpt 2>/dev/null))
[ ${#CK[@]} -gt 0 ] || { echo "no ms checkpoints yet"; exit 0; }
echo "${#CK[@]} checkpoints"
HELDOUT_TRAIN_SHARDS="00787 00000" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
