#!/bin/bash
# Submit GMFlow on the full ladder (docs/gmflow.md section 17, owner 2026-11-06 21:4x popup: GMFlow on the full ladder) to wq, first come first served (prio 0):
# K = 8 per pixel at d640 to 1C0, d768 to 2C0, d896 to 4C0, d1024 to 8C0, the point model's widths, data, shard seed, segments and evaluation points. Per size and
# segment: one training job (jobs/train_full_gmflow.sh, --partial, after ok of the size's segment before it) and one evaluation job (gpic-wt/gmflow_run4/jobs/
# evals_gmflow.sh, the code of the point model's full-ladder evaluations and of the Lite GMFlow evaluations: GM-ODE 8, 50 and 4 calls and Euler on the mixture
# mean at 50 calls, at C0/4, C0/2 and C0 for the 1C0 segment and at the segment's end otherwise; official noise, EMA, CFG = 1; after ok of its training job).
# Order as tools/submit_full_ladder.sh: segment by segment, each segment's evaluations right after its trainings.
# C0 steps: the largest multiple of 800 with steps x (1 + r) <= the point model's C0 steps (r = the head's added forward FLOPs / the forward FLOPs per image;
# tmp/ladder_prof/gm_full_steps.py reproduces the Lite step counts 499328, 322688, 221056, 155520 with the same r).
# Time limits: training 1.25 x hours per C0 + 15 min, the hours being the point model's measured 1C0 segments (w1326-w1329: 3.95, 3.15, 2.83, 2.52 h) x 1.05;
# evaluation 1.3 x checkpoints x (15 min + 112 network calls at 1.07 s per call at d640, scaled by the forward GFLOP; the Lite d640 evaluations took 4.4-5.1 min per
# task), at least 45 min.
# usage (login node, in the frozen worktree gpic-wt/ladder_full_gmflow_run): bash tools/submit_full_gmflow.sh [--dry]
# Writes logs/full_ladder_gmflow_jobs.tsv: size, multiple, training job, evaluation job, steps, hours.
set -euo pipefail
WC=/scratch/gpfs/GROUP/USER
WQ=$WC/tools/wq/wq
WT=${FULL_WT:-$WC/project/gpic-wt/ladder_full_gmflow_run}   # FULL_WT: a dry run in another worktree
EV=$WC/project/gpic-wt/gmflow_run4
K=8
ROOT=$WC/runs/gpic-speed/full_ladder_gmflow_k$K
DRY=${1:-}
[ "$(pwd -P)" = "$(cd $WT && pwd -P)" ] || { echo "run in $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
[ -z "$(git -C $EV status --porcelain --untracked-files=no)" ] || { echo "$EV has local changes"; exit 2; }
# size d nb nt C0-steps max-multiple hours-per-C0-on-8-H100 forward-GFLOP-per-image
SIZES="d640 640 9 2 499200 1 4.15 34.7
d768 768 10 2 322400 2 3.31 54.6
d896 896 11 2 220800 4 2.97 80.7
d1024 1024 12 3 155200 8 2.65 115.6"
hm() { awk -v h="$1" 'BEGIN { q = int(h * 4 + 0.999); printf "%02d:%02d:00", int(q / 4), (q % 4) * 15 }'; }   # hours, up to the next quarter
calc() { awk "BEGIN { printf \"%.3f\", $1 }"; }
submit() {   # submit <dir> <wq submit arguments...>: prints the new job id
  local dir=$1; shift
  if [ -n "$DRY" ]; then echo "DRY (cd $dir) wq submit $*" >&2; echo "w0"; return; fi
  (cd $dir && $WQ submit "$@" | tail -1)
}
OUT=$WT/logs/full_ladder_gmflow_jobs.tsv
if [ -z "$DRY" ]; then mkdir -p $WT/logs; printf "size\tmultiple\ttrain\teval\tsteps\ttrain_h\teval_h\n" > $OUT; fi
declare -A PREV TRAIN
prev_m=0
for M in 1 2 4 8; do
  while read -r S D NB NT C0S MAX H G; do
    [ $M -le $MAX ] || continue
    th=$(calc "1.25 * ($M - $prev_m) * $H + 0.25")
    AFTER=(); [ -z "${PREV[$S]:-}" ] || AFTER=(--after ok:${PREV[$S]})
    TRAIN[$S]=$(submit $WT --partial --time $(hm $th) -J gpic-full-gmflow-k$K-$S-${M}c0 --project gpic-wt "${AFTER[@]}" \
      jobs/train_full_gmflow.sh $K $S $D $NB $NT $C0S $M)
    PREV[$S]=${TRAIN[$S]}
    echo "$S ${M}C0: training ${TRAIN[$S]}, $((C0S * M)) steps, $th h"
  done <<< "$SIZES"
  while read -r S D NB NT C0S MAX H G; do
    [ $M -le $MAX ] || continue
    if [ $M -eq 1 ]; then STEPS="$((C0S / 4)) $((C0S / 2)) $C0S"; else STEPS="$((C0S * M))"; fi
    TASKS=(); n=0
    for st in $STEPS; do TASKS+=($st:gm_ode:8 $st:gm_ode:50 $st:gm_ode:4 $st:euler_mean:50); n=$((n + 1)); done
    eh=$(awk -v n=$n -v g=$G 'BEGIN { x = 1.3 * n * (15 + 112 * 1.07 * g / 34.7 / 60) / 60; if (x < 0.75) x = 0.75; printf "%.3f", x }')
    EJ=$(submit $EV --time $(hm $eh) -J gpic-full-gmflow-evals-k$K-$S-${M}c0 --project gpic-wt --after ok:${TRAIN[$S]} \
      jobs/evals_gmflow.sh $K $S $D $NB $NT $ROOT --wt gmflow_run4 "${TASKS[@]}")
    echo "$S ${M}C0: evaluation $EJ at $STEPS, $eh h"
    [ -n "$DRY" ] || printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $S $M ${TRAIN[$S]} $EJ $((C0S * M)) $(calc "1.25 * ($M - $prev_m) * $H + 0.25") $eh >> $OUT
  done <<< "$SIZES"
  prev_m=$M
done
