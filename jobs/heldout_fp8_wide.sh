#!/bin/bash
#SBATCH -J gpic-heldout-fp8wide
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss of the wide-width FP8 check (owner popup 2026-11-07 19:4x): the ms checkpoints of runs/gpic-speed/muon_fp8check/
# exp_ladder_d1280 (FP8 native3) and of its bf16 reference, the Full ladder's d1280 at the same steps (5600, 11200, 22400), EMA and
# raw, N 3072 + 3072, appended to tmp/heldout_loss/heldout_loss.jsonl (idempotent). Code: the frozen copy heldout_loss_run @ 7449b49,
# as jobs/heldout_bucket_ab.sh.
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss_run
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=($(ls $WC/runs/gpic-speed/muon_fp8check/exp_ladder_d1280/ms-step=*.ckpt $WC/runs/gpic-speed/muon_full/exp_ladder_d1280/ms-step=0{05600,11200,22400}.ckpt 2>/dev/null))
[ ${#CK[@]} -gt 0 ] || { echo "no ms checkpoints yet"; exit 0; }
echo "${#CK[@]} checkpoints"
HELDOUT_TRAIN_SHARDS="00787 00000" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
