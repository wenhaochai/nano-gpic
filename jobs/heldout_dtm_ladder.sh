#!/bin/bash
#SBATCH -J gpic-dtm-ladder-heldout
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:30:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The held-out loss of the final ladder's DTM flow-head runs (jobs/train_dtm_ladder.sh; docs/dtm_ladder.md), as tmp/ladder_prof/wsd/heldout_sweep.sh scores the point runs:
# the final checkpoint of every finished run in <root>/dtm_ladder that has no record yet, with the ladder's tool on the ladder's data (3072 held-out and 3072
# training images, 16 noise levels, raw weights; tools/heldout_loss.py). The head is scored by its mean (the model's default output: the head's x prediction at
# head time 0, averaged over 4 fixed head noises), the same quantity as the point model's loss. One checkpoint per GPU on a shared counter.
# usage: heldout_dtm_ladder.sh [--root <runs root>] [--out <records.jsonl>]
#   --root: another runs/gpic-speed; --out: another record file than tmp/heldout_loss/heldout_loss_dtm.jsonl (both: the smoke test).
# Code: the frozen worktree gpic-wt/dtm_ladder_run.
export NCCL_NVLS_ENABLE=0   # as tmp/ladder_prof/wsd/heldout_sweep.sh
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/dtm_ladder_run
source $WC/project/gpic/jobs/env.sh
R=$WC/runs/gpic-speed OUTF=$WC/tmp/heldout_loss/heldout_loss_dtm.jsonl
while [ $# -gt 0 ]; do
  case "$1" in
    --root) R=$2; shift 2; [ -d "$R" ] || { echo "no root $R"; exit 2; } ;;
    --out) OUTF=$2; shift 2 ;;
    *) echo "usage: heldout_dtm_ladder.sh [--root <runs root>] [--out <records.jsonl>]"; exit 2 ;;
  esac
done
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "dtm_width" $WT/tools/heldout_loss.py || { echo "$WT has no DTM support in tools/heldout_loss.py"; exit 2; }
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=()
for d in $R/dtm_ladder/*/exp_ladder_*; do   # only finished runs: the checkpoint of the run's max_steps (launches.txt), never a run in progress
  [ -f "$d/launches.txt" ] || continue
  S=$(grep -o "max_steps=[0-9]*" $d/launches.txt | tail -1 | cut -d= -f2) || true
  [[ $S =~ ^[1-9][0-9]*$ ]] || { echo "$d: no max_steps in launches.txt; left out"; continue; }
  f=$d/ms-step=$(printf %06d $S).ckpt
  if [ -f "$f" ]; then CK+=("$f"); else echo "$d: not finished (no ms-step=$(printf %06d $S).ckpt); left out"; fi
done
echo "${#CK[@]} final checkpoints under $R/dtm_ladder"
[ ${#CK[@]} -gt 0 ] || { echo "no final checkpoint"; exit 1; }
HELDOUT_TRAIN_SHARDS="00787" heldout_run $OUTF 3072 3072 4 "${CK[@]}"
