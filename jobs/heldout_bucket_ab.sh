#!/bin/bash
#SBATCH -J gpic-heldout-search
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:15:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss of the shared-bucket A/B (2026-11-07 06:5x; muon-search2 jobs/train_bucket_ab.sh): every ms checkpoint under
# runs/gpic-speed/muon_bucket_ab/<arm>/exp_ladder_d640 and the Full ladder's width-640 checkpoint at 33400 (the shared-bucket arm),
# EMA and raw, N 3072 + 3072, appended to tmp/heldout_loss/heldout_loss.jsonl (idempotent). Code: the frozen copy heldout_loss_run @ 7449b49.
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss_run   # frozen at 7449b49
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=($(ls $WC/runs/gpic-speed/muon_bucket_ab/*/exp_ladder_d640/ms-step=*.ckpt $WC/runs/gpic-speed/muon_full/exp_ladder_d640/ms-step=033400.ckpt 2>/dev/null))
[ ${#CK[@]} -gt 0 ] || { echo "no ms checkpoints yet"; exit 0; }
echo "${#CK[@]} checkpoints"
HELDOUT_TRAIN_SHARDS="00787 00000" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
