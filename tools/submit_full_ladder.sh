#!/bin/bash
# Submit the full ladder's point-model runs (docs/ladder_plan.md section 12, owner 2026-11-06) to wq, first come first
# served (prio 0). Per size and budget segment: one training job (jobs/train_full.sh, --partial, after ok of the size's
# segment before it; the first segment after ok of the text cache job) and one evaluation job (gpic-wt/gmflow_run4/jobs/
# evals_gmflow.sh, Euler 8, 50 and 4 calls at C0/4, C0/2 and C0 for the 1C0 segment and at the segment's end otherwise;
# official noise, EMA, CFG = 1; after ok of its training job). Order: segment by segment, each segment's evaluations right
# after its trainings, so a ready evaluation starts before the next segment's training (wq takes ready jobs by id).
# Time limits: training 1.25 x the measured hours per C0 + 15 min; evaluation 1.3 x (15 min per checkpoint + the sampling
# at 300 TFLOP/s per GPU), at least 45 min (w1036: one 50-call eval of d1280 took 0.13 h on 8 GPUs).
# usage (login node, in the frozen worktree gpic-wt/ladder_full_run): bash tools/submit_full_ladder.sh <cache job> [--dry]
# Writes logs/full_ladder_jobs.tsv: size, multiple, training job, evaluation job, steps, hours.
set -euo pipefail
WC=/scratch/gpfs/GROUP/USER
WQ=$WC/tools/wq/wq
WT=${FULL_WT:-$WC/project/gpic-wt/ladder_full_run}   # FULL_WT: a dry run in another worktree
EV=$WC/project/gpic-wt/gmflow_run4
ROOT=$WC/runs/gpic-speed/full_ladder
[ $# -ge 1 ] || { echo "usage: submit_full_ladder.sh <cache job> [--dry]"; exit 2; }
CACHE=$1 DRY=${2:-}
[ "$(pwd -P)" = "$(cd $WT && pwd -P)" ] || { echo "run in $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
[ -z "$(git -C $EV status --porcelain --untracked-files=no)" ] || { echo "$EV has local changes"; exit 2; }
# size d nb nt C0-steps max-multiple hours-per-C0-on-8-H100 forward-GFLOP-per-image
SIZES="d640 640 9 2 534400 1 4.85 34.7
d768 768 10 2 340000 2 3.37 54.6
d896 896 11 2 230400 4 2.97 80.7
d1024 1024 12 3 160800 8 2.62 115.6
d1280 1280 14 3 89600 8 2.36 207.5
d1536 1536 16 4 54400 8 H1536 341.6
d1792 1792 18 4 36000 8 H1792 518.3"
if grep -q -E "H1536|H1792" <<< "$SIZES"; then echo "fill in the measured hours per C0 of d1536 and d1792 (smoke w1313)"; [ -n "$DRY" ] || exit 2; SIZES=${SIZES//H1536/2.36}; SIZES=${SIZES//H1792/2.36}; fi
hm() { awk -v h="$1" 'BEGIN { q = int(h * 4 + 0.999); printf "%02d:%02d:00", int(q / 4), (q % 4) * 15 }'; }   # hours, up to the next quarter
calc() { awk "BEGIN { printf \"%.3f\", $1 }"; }
submit() {   # submit <dir> <wq submit arguments...>: prints the new job id
  local dir=$1; shift
  if [ -n "$DRY" ]; then echo "DRY (cd $dir) wq submit $*" >&2; echo "w0"; return; fi
  (cd $dir && $WQ submit "$@" | tail -1)
}
OUT=$WT/logs/full_ladder_jobs.tsv
if [ -z "$DRY" ]; then mkdir -p $WT/logs; printf "size\tmultiple\ttrain\teval\tsteps\ttrain_h\teval_h\n" > $OUT; fi
declare -A PREV TRAIN
prev_m=0
for M in 1 2 4 8; do
  while read -r S D NB NT C0S MAX H G; do
    [ $M -le $MAX ] || continue
    th=$(calc "1.25 * ($M - $prev_m) * $H + 0.25")
    TRAIN[$S]=$(submit $WT --partial --time $(hm $th) -J gpic-full-$S-${M}c0 --project gpic-wt --after ok:${PREV[$S]:-$CACHE} \
      jobs/train_full.sh $S $D $NB $NT $C0S $M)
    PREV[$S]=${TRAIN[$S]}
    echo "$S ${M}C0: training ${TRAIN[$S]}, $((C0S * M)) steps, $th h"
  done <<< "$SIZES"
  while read -r S D NB NT C0S MAX H G; do
    [ $M -le $MAX ] || continue
    if [ $M -eq 1 ]; then STEPS="$((C0S / 4)) $((C0S / 2)) $C0S"; else STEPS="$((C0S * M))"; fi
    TASKS=(); n=0
    for st in $STEPS; do TASKS+=($st:euler:8 $st:euler:50 $st:euler:4); n=$((n + 1)); done
    eh=$(awk -v n=$n -v g=$G 'BEGIN { x = 1.3 * n * (15 + 1.29 * g / 60) / 60; if (x < 0.75) x = 0.75; printf "%.3f", x }')
    EJ=$(submit $EV --time $(hm $eh) -J gpic-full-evals-$S-${M}c0 --project gpic-wt --after ok:${TRAIN[$S]} \
      jobs/evals_gmflow.sh 0 $S $D $NB $NT $ROOT --wt gmflow_run4 "${TASKS[@]}")
    echo "$S ${M}C0: evaluation $EJ at $STEPS, $eh h"
    [ -n "$DRY" ] || printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $S $M ${TRAIN[$S]} $EJ $((C0S * M)) $(calc "1.25 * ($M - $prev_m) * $H + 0.25") $eh >> $OUT
  done <<< "$SIZES"
  prev_m=$M
done
