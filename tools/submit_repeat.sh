#!/bin/bash
# Submit the repetition arms (owner 2026-11-07) to wq: d640 and d1024 on the first 80 and the first 8 shards of GPIC train, each to C0
# (jobs/train_repeat.sh, --partial), and after ok of each training one evaluation job (gpic-wt/gmflow_run4/jobs/evals_gmflow.sh, K = 0,
# Euler at 50 calls at C0/4, C0/2 and C0; official noise, EMA, CFG = 1). Prio 15 for all (owner: the ladder is the focus).
# Time limits: training 1.25 x the measured hours per C0 of tools/submit_full_ladder.sh + 15 min (the arms run the same step);
# evaluation as in submit_full_ladder.sh: 1.3 x (15 min per checkpoint + the sampling at 300 TFLOP/s per GPU), at least 45 min.
# usage (login node, in the frozen worktree gpic-wt/repeat_run): bash tools/submit_repeat.sh [--dry]
# Writes logs/repeat_jobs.tsv: shards, size, training job, evaluation job, steps, hours.
set -euo pipefail
WC=/scratch/gpfs/GROUP/USER
WQ=$WC/tools/wq/wq
WT=${REPEAT_WT:-$WC/project/gpic-wt/repeat_run}   # REPEAT_WT: a dry run in another worktree
EV=$WC/project/gpic-wt/gmflow_run4
DRY=${1:-}
case $DRY in ""|--dry) ;; *) echo "usage: submit_repeat.sh [--dry]"; exit 2 ;; esac
[ "$(pwd -P)" = "$(cd $WT && pwd -P)" ] || { echo "run in $WT"; exit 2; }
if [ -z "$DRY" ]; then   # a dry run may show an uncommitted tree
  [ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
fi
[ -z "$(git -C $EV status --porcelain --untracked-files=no)" ] || { echo "$EV has local changes"; exit 2; }
grep -q "def _repeat_copies" $WT/baselines/PixelGen/src/data/dataset/gpic_dataset.py || { echo "$WT has no shard copies"; exit 2; }
# size d nb nt C0-steps hours-per-C0-on-8-H100 forward-GFLOP-per-image (tools/submit_full_ladder.sh)
SIZES="d640 640 9 2 534400 4.85 34.7
d1024 1024 12 3 160800 2.62 115.6"
hm() { awk -v h="$1" 'BEGIN { q = int(h * 4 + 0.999); printf "%02d:%02d:00", int(q / 4), (q % 4) * 15 }'; }   # hours, up to the next quarter
submit() {   # submit <dir> <wq submit arguments...>: prints the new job id
  local dir=$1; shift
  if [ -n "$DRY" ]; then echo "DRY (cd $dir) wq submit $*" >&2; echo "w0"; return; fi
  (cd $dir && $WQ submit "$@" | tail -1)
}
OUT=$WT/logs/repeat_jobs.tsv
if [ -z "$DRY" ]; then mkdir -p $WT/logs; printf "shards\tsize\ttrain\teval\tsteps\ttrain_h\teval_h\n" > $OUT; fi
for N in 80 8; do
  while read -r S D NB NT C0S H G; do
    th=$(awk -v h=$H 'BEGIN { printf "%.3f", 1.25 * h + 0.25 }')
    TJ=$(submit $WT --partial --prio 15 --time $(hm $th) -J gpic-repeat-s$N-$S --project gpic-wt \
      jobs/train_repeat.sh $S $D $NB $NT $C0S $N)
    echo "s$N $S: training $TJ, $C0S steps, $th h"
    eh=$(awk -v g=$G 'BEGIN { x = 1.3 * 3 * (15 + 1.29 * g / 60) / 60; if (x < 0.75) x = 0.75; printf "%.3f", x }')
    EJ=$(submit $EV --prio 15 --time $(hm $eh) -J gpic-repeat-evals-s$N-$S --project gpic-wt --after ok:$TJ \
      jobs/evals_gmflow.sh 0 $S $D $NB $NT $WC/runs/gpic-speed/repeat_s$N --wt gmflow_run4 \
      $((C0S / 4)):euler:50 $((C0S / 2)):euler:50 $C0S:euler:50)
    echo "s$N $S: evaluation $EJ at $((C0S / 4)) $((C0S / 2)) $C0S, $eh h"
    [ -n "$DRY" ] || printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $N $S $TJ $EJ $C0S $th $eh >> $OUT
  done <<< "$SIZES"
done
