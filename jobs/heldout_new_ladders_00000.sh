#!/bin/bash
#SBATCH -J gpic-heldout-search
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:15:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Held-out loss of the new ladders on the second train split only (gpic_train_00000: seen by s80 and s8 from step 0, by Lite from
# about step 6,200, by Full only from about step 75,000; tmp/ladder_prof/heldout_v2_impl.md section 5): every ms checkpoint under
# runs/gpic-speed/muon_{full,lite,s80,s8}, EMA and raw, appended to tmp/heldout_loss/heldout_loss.jsonl, keyed by the 00000 data file
# (idempotent). The 00787 split of the same checkpoints comes from jobs/heldout_new_ladders.sh. Code: the frozen copy
# gpic-wt/heldout_loss_run2 @ 30e02cb (the --train-shard option), so edits in this worktree do not reach a queued job.
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_loss_run2   # frozen at 30e02cb
source $WC/project/gpic/jobs/env.sh
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=($(ls $WC/runs/gpic-speed/muon_{full,lite,s80,s8}/exp_ladder_d*/ms-step=*.ckpt 2>/dev/null))
[ ${#CK[@]} -gt 0 ] || { echo "no ms checkpoints yet"; exit 0; }
echo "${#CK[@]} checkpoints"
HELDOUT_TRAIN_SHARDS="00000" heldout_run $WC/tmp/heldout_loss/heldout_loss.jsonl 3072 3072 8 "${CK[@]}"
