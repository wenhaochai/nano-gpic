#!/bin/bash
#SBATCH -J gpic-heldout-tuniform
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:40:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# The held-out loss at 16 levels uniform in t (tools/heldout_loss.py --levels uniform: t = (k + 0.5) / 16) for a list of checkpoints, with the
# ladder's tool, data and noise (3072 held-out and 3072 training images). Owner 2026-11-12 23:5x 「好」 to a direct measurement of the question
# "does another timestep distribution of the held-out loss change the scaling law": the tool's default levels are the quantiles of the training
# distribution and cover only t = 0.09 to 0.67 (tmp/ladder_prof/tdist/README.md). One checkpoint per GPU on a shared counter; finished records
# are skipped, so a cut job continues where it stopped (exit 0 only when no task is left).
# usage: heldout_tuniform.sh <file with one checkpoint path per line> [<records.jsonl>]
#   records: tmp/heldout_loss/heldout_loss_tuniform.jsonl by default (never the ladder's heldout_loss.jsonl: its readers take the mean of the levels
#   as the training-distribution loss).
# Code: the frozen worktree gpic-wt/heldout_tlevels_run.
export NCCL_NVLS_ENABLE=0   # as tmp/ladder_prof/wsd/heldout_sweep.sh
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/heldout_tlevels_run
source $WC/project/gpic/jobs/env.sh
LIST=${1:-} OUTF=${2:-$WC/tmp/heldout_loss/heldout_loss_tuniform.jsonl}
[ -s "$LIST" ] || { echo "usage: heldout_tuniform.sh <file with one checkpoint path per line> [<records.jsonl>]"; exit 2; }
case "$OUTF" in */heldout_loss.jsonl) echo "the uniform-level records need their own file, not $OUTF"; exit 2 ;; esac
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q -- '"--levels"' $WT/tools/heldout_loss.py || { echo "$WT has no --levels in tools/heldout_loss.py"; exit 2; }
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
mapfile -t CK < <(grep -v '^#' "$LIST" | awk 'NF')
[ ${#CK[@]} -gt 0 ] || { echo "no checkpoint in $LIST"; exit 2; }
for f in "${CK[@]}"; do [ -f "$f" ] || { echo "no checkpoint $f"; exit 2; }; done
echo "${#CK[@]} checkpoints from $LIST; records in $OUTF"
HELDOUT_LEVELS=uniform HELDOUT_TRAIN_SHARDS="00787" heldout_run $OUTF 3072 3072 4 "${CK[@]}"
