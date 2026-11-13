#!/bin/bash
#SBATCH -J gpic-gm-ladder-heldout
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:30:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The held-out loss of the final ladder's mixture runs (jobs/train_gmflow_ladder.sh), as tmp/ladder_prof/wsd/heldout_sweep.sh scores the ladder's point runs:
# the final checkpoint of every finished run in <root>/gm_ladder that has no record yet, with the ladder's tool on the ladder's data (3072 held-out and 3072
# training images, 16 noise levels, raw weights; tools/heldout_loss.py). The mixture is scored by its mean, the same quantity as the point model's loss,
# and with --nll by its negative log-likelihood per dimension at each level, beside 0.5 log(2 pi e MSE), the best fixed-variance Gaussian around the mean.
# One checkpoint per GPU on a shared counter (4 GPUs: the four runs of two budgets leave no GPU idle).
# usage: heldout_gm_ladder.sh [--root <runs root>] [--out <records.jsonl>]
#   --root: another runs/gpic-speed; --out: another record file than tmp/heldout_loss/heldout_loss_gm.jsonl (both: the smoke test).
# Code: the frozen worktree gpic-wt/gmflow_ladder_run.
export NCCL_NVLS_ENABLE=0   # as tmp/ladder_prof/wsd/heldout_sweep.sh
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/gmflow_ladder_run
source $WC/project/gpic/jobs/env.sh
R=$WC/runs/gpic-speed OUTF=$WC/tmp/heldout_loss/heldout_loss_gm.jsonl
while [ $# -gt 0 ]; do
  case "$1" in
    --root) R=$2; shift 2; [ -d "$R" ] || { echo "no root $R"; exit 2; } ;;
    --out) OUTF=$2; shift 2 ;;
    *) echo "usage: heldout_gm_ladder.sh [--root <runs root>] [--out <records.jsonl>]"; exit 2 ;;
  esac
done
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q -- "--nll" $WT/tools/heldout_loss.py || { echo "$WT has no --nll in tools/heldout_loss.py"; exit 2; }
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=()
for d in $R/gm_ladder/*/exp_ladder_*; do   # only finished runs: the checkpoint of the run's max_steps (launches.txt), never a run in progress
  [ -f "$d/launches.txt" ] || continue
  S=$(grep -o "max_steps=[0-9]*" $d/launches.txt | tail -1 | cut -d= -f2) || true
  [[ $S =~ ^[1-9][0-9]*$ ]] || { echo "$d: no max_steps in launches.txt; left out"; continue; }
  f=$d/ms-step=$(printf %06d $S).ckpt
  if [ -f "$f" ]; then CK+=("$f"); else echo "$d: not finished (no ms-step=$(printf %06d $S).ckpt); left out"; fi
done
echo "${#CK[@]} final checkpoints under $R/gm_ladder"
[ ${#CK[@]} -gt 0 ] || { echo "no final checkpoint"; exit 1; }
HELDOUT_NLL=1 HELDOUT_TRAIN_SHARDS="00787" heldout_run $OUTF 3072 3072 4 "${CK[@]}"
