#!/bin/bash
#SBATCH -J gpic-point-calls
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The final ladder's point runs at another number of Euler calls (owner popup 2026-11-13 00:5x: the DTM flow head and the point model at 50 and 16 calls;
# docs/dtm_ladder.md), with the code of the ladder's own evaluations (gpic-wt/eval_official_run11: jobs/evals_gmflow.sh K = 0, raw-weight views, official
# noise, no CFG, 50K captions, the DINOv2 and the Inception metrics), so the result sits beside the ladder's 50-call one:
# runs/gpic-speed/raw_view/<rel>/exp_ladder_<size>/eval_<step>_euler_calls<calls>_official. Finished tasks are skipped.
# usage: evals_point_calls.sh <calls> <rel>:<size>:<d>:<nb>:<nt>:<step> ...   (the raw view must exist: the ladder's evaluation made it)
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
source $WC/project/gpic/jobs/env.sh
WTN=eval_official_run11
WT=$WC/project/gpic-wt/$WTN
CALLS=$1; shift
[[ $CALLS =~ ^[1-9][0-9]*$ ]] && [ $# -ge 1 ] || { echo "usage: evals_point_calls.sh <calls> <rel>:<size>:<d>:<nb>:<nt>:<step> ..."; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
R=$WC/runs/gpic-speed fail=0
for t in "$@"; do
  IFS=: read -r rel sz d nb nt st extra <<< "$t"
  [[ -z "$extra" && $rel =~ ^[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*$ && $sz =~ ^[A-Za-z0-9_-]+$ && $d =~ ^[0-9]+$ && $nb =~ ^[0-9]+$ && $nt =~ ^[0-9]+$ && $st =~ ^[0-9]+$ ]] \
    || { echo "task $t is not <rel>:<size>:<d>:<nb>:<nt>:<step>"; exit 2; }
  [ -e $R/raw_view/$rel/exp_ladder_$sz/ms-step=$(printf %06d $st).ckpt ] || { echo "task $t: no raw view"; fail=1; continue; }
  bash $WT/jobs/evals_gmflow.sh 0 $sz $d $nb $nt $R/raw_view/$rel --wt $WTN $st:euler:$CALLS || { echo "task $t failed"; fail=1; }
  f=$R/raw_view/$rel/exp_ladder_$sz/eval_$(printf %06d $st)_euler_calls${CALLS}_official
  [ -s $f/metrics.json ] && [ -s $f/metrics_inception.json ] && echo "task $t: $(cat $f/metrics.json | tr -d '\n ' | cut -c1-80)" || { echo "task $t: no metrics"; fail=1; }
done
exit $fail
