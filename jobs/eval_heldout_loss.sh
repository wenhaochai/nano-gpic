#!/bin/bash
#SBATCH -J gpic-heldout-loss
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out and train loss of every ms checkpoint of the point-model ladders (tools/heldout_loss.py; owner 2026-11-06):
# Lite runs/gpic-speed/exp_ladder_<size>, ladder_<size>_<k>c0/exp_ladder_<size>, Full runs/gpic-speed/full_ladder/exp_ladder_<size>.
# EMA and raw weights; N_VAL held-out images (gpic_val_*.tar) and N_TRAIN train images (gpic_train_00787.tar), 16 levels.
# The task list holds every checkpoint older than 10 min that lacks a line in OUT; 8 workers, one per GPU, take the next
# task from a shared lockf counter, longest first, so no GPU idles while tasks remain. Idempotent: a rerun computes only
# what is missing, so a stopped job loses at most the checkpoints in progress. Exit 0 when every listed task has its lines.
# Estimate per checkpoint on one H100 (A40 measurements / 4.8, tools/heldout_loss.py est_minutes): d640 1.6 min, d1024
# 3.0, d1280 4.2, d1792 7.9; the 49 checkpoints of 2026-11-07 take about 116 GPU-min, 15 min on 8 GPUs (longest first).
# jobs/smoke_heldout_loss.sh measures the H100. The data file (tmp/heldout_loss/data_v1_val3072_train3072_00787.pt) exists.
# Writes only under tmp/heldout_loss/ (never into runs/).
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss   # the worktree whose code runs (a wq job cannot take it from the submit shell)
N_VAL=3072 N_TRAIN=3072 NGPU=8
OUT=$WC/tmp/heldout_loss/heldout_loss.jsonl
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
heldout_run "$OUT" $N_VAL $N_TRAIN $NGPU
