#!/bin/bash
#SBATCH -J gpic-ddm-ladder-heldout
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:30:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The held-out scores of the final ladder's distributional-denoiser runs (jobs/train_ddm_ladder.sh, docs/ddm.md), as jobs/heldout_gm_ladder.sh scores the
# mixture: the final checkpoint of every finished run in <root>/ddm_ladder that has no record yet, with the ladder's tool on the ladder's data (3072 held-out
# and 3072 training images, 16 noise levels, raw weights; tools/heldout_loss.py --es). A run is scored by 4 draws per image and level: the squared error of
# the draws' mean with their variance removed (the point model's loss, estimated without bias) and the energy score per token. The point run of the same
# budget gets its energy score too (POINT below), so the two models compare on both numbers. One checkpoint per GPU on a shared counter.
# usage: heldout_ddm_ladder.sh [--root <runs root>] [--out <records.jsonl>] [--no-point]
#   --root: another runs/gpic-speed; --out: another record file than tmp/heldout_loss/heldout_loss_ddm.jsonl (both: the smoke test); --no-point: only the
#   ddm runs (the smoke test).
# Code: the frozen worktree gpic-wt/ddm_run.
export NCCL_NVLS_ENABLE=0   # as tmp/ladder_prof/wsd/heldout_sweep.sh
set -eo pipefail
WC=/scratch/gpfs/GROUP/USER
WT=$WC/project/gpic-wt/ddm_run
source $WC/project/gpic/jobs/env.sh
R=$WC/runs/gpic-speed OUTF=$WC/tmp/heldout_loss/heldout_loss_ddm.jsonl NOPOINT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --root) R=$2; shift 2; [ -d "$R" ] || { echo "no root $R"; exit 2; } ;;
    --out) OUTF=$2; shift 2 ;;
    --no-point) NOPOINT=1; shift ;;
    *) echo "usage: heldout_ddm_ladder.sh [--root <runs root>] [--out <records.jsonl>] [--no-point]"; exit 2 ;;
  esac
done
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q -- '"--es"' $WT/tools/heldout_loss.py || { echo "$WT has no --es in tools/heldout_loss.py"; exit 2; }
export PYTHONPATH=$WT/baselines/PixelGen OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source $WT/jobs/heldout_loss_lib.sh
CK=()
declare -A POINT=([exp_ladder_d768]=muon_sweep/xl-d10-0p25c0/exp_ladder_d768/ms-step=021248.ckpt
                  [exp_ladder_d1024]=muon_sweep/xl-d12-1c0/exp_ladder_d1024/ms-step=040192.ckpt)   # the point runs of the same budgets (tools/ddm_ladder/steps.py)
PT=()
for d in $R/ddm_ladder/*/exp_ladder_*; do   # only finished runs: the checkpoint of the run's max_steps (launches.txt), never a run in progress
  [ -f "$d/launches.txt" ] || continue
  S=$(grep -o "max_steps=[0-9]*" $d/launches.txt | tail -1 | cut -d= -f2) || true
  [[ $S =~ ^[1-9][0-9]*$ ]] || { echo "$d: no max_steps in launches.txt; left out"; continue; }
  f=$d/ms-step=$(printf %06d $S).ckpt
  if [ -f "$f" ]; then CK+=("$f"); else echo "$d: not finished (no ms-step=$(printf %06d $S).ckpt); left out"; fi
  p=${POINT[$(basename $d)]:-}
  if [ $NOPOINT = 0 ] && [ -n "$p" ]; then
    [ -f $WC/runs/gpic-speed/$p ] || { echo "no point checkpoint $WC/runs/gpic-speed/$p"; exit 2; }
    case " ${PT[*]} " in *" $WC/runs/gpic-speed/$p "*) ;; *) PT+=($WC/runs/gpic-speed/$p) ;; esac
  fi
done
echo "${#CK[@]} final checkpoints under $R/ddm_ladder; ${#PT[@]} point checkpoints"
[ ${#CK[@]} -gt 0 ] || { echo "no final checkpoint"; exit 1; }
HELDOUT_ES=1 HELDOUT_TRAIN_SHARDS="00787" heldout_run $OUTF 3072 3072 4 "${CK[@]}" "${PT[@]}"
