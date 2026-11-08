#!/bin/bash
#SBATCH -J gpic-heldout-fp8recipe
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss (raw weights) of every ms checkpoint of the d20 FP8 check's arms (gpic-28; owner popups 2026-11-08 03:4x and 11:2x):
# genmodel's tmp/ladder_prof/wsd/heldout_sweep_done.sh with this check's checkpoints in place of the sweep list. Lines go to
# tmp/heldout_loss/heldout_loss.jsonl; checkpoints done before are skipped.
# usage: heldout_fp8recipe.sh <size> <tag> [<tag> ...]   (runs/gpic-speed/muon_fp8recipe/<tag>/exp_ladder_<size>/ms-step=*.ckpt)
export NCCL_NVLS_ENABLE=0   # NCCL 2.26.2 NVLS all-gather can permute rank blocks (NVIDIA/nccl #1906; gpic-28 w2090)
set -eo pipefail
shopt -s nullglob
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss_run
SIZE=${1:?size}; shift
[ $# -ge 1 ] || { echo "usage: heldout_fp8recipe.sh <size> <tag>..."; exit 2; }
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=$WC/tmp/job/${FETCHED_JOB_ID:-$SLURM_JOB_ID}; mkdir -p $TMPDIR
source $WT/jobs/heldout_loss_lib.sh
CK=()
for tag in "$@"; do CK+=($WC/runs/gpic-speed/muon_fp8recipe/$tag/exp_ladder_$SIZE/ms-step=*.ckpt); done
[ ${#CK[@]} -gt 0 ] || { echo "no ms checkpoints"; exit 1; }
echo "${#CK[@]} checkpoints"
HELDOUT_TRAIN_SHARDS="00787" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
