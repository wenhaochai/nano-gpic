#!/bin/bash
#SBATCH -J gpic-gm-ladder-evals
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The evaluations of the final ladder's mixture runs (jobs/train_gmflow_ladder.sh), as jobs/evals_group.sh evaluates the ladder's point runs: the raw-weight
# view of the checkpoint (tools/make_raw_view.py: a link, since a no-EMA checkpoint holds the raw weights in its EMA slot), then jobs/evals_gmflow.sh on the
# view: 50K test captions, the official noise, no CFG, 50 calls, the DINOv2 and the Inception metrics. Three samplers per run (owner popups 2026-11-12 22:5x):
#   gm_draw     at every call a draw from the mixture by its probabilities, then a straight step (the main result; the owner's sampler of 2026-11-06);
#   gm_sde      the same draw plus fresh noise (reference);
#   euler_mean  Euler on the mixture mean (control).
# All tasks of the job form one chain (jobs/eval_chain_lib.sh: a task's scoring runs beside the next task's predict). Finished tasks are skipped.
# usage: evals_gm_ladder.sh [--root <runs root>] [--limit <batches>] [--samplers "<s> ..."] <tag>:<d>:<nb>:<nt>:<step> ...
#   tag = gm<K>-<name>, K = 8 or 1: the run is <root>/gm_ladder/<tag>/exp_ladder_d<d>, its results are in
#   <root>/raw_view/gm_ladder/<tag>/exp_ladder_d<d>/eval_<step>_<sampler>_calls50_official/{metrics.json,metrics_inception.json}.
#   --root: another runs/gpic-speed (the smoke test). --limit: predict only that many batches per rank and check the samples; no metrics, no chain (the smoke
#   test). --samplers: a subset of the three (the smoke test).
# Code: the frozen worktree gpic-wt/gmflow_ladder_run.
export NCCL_NVLS_ENABLE=0   # NCCL 2.26.2 NVLS all-gather can permute rank blocks (NVIDIA/nccl #1906); as jobs/evals_group.sh
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
source $WC/project/gpic/jobs/env.sh
WTN=gmflow_ladder_run
R=$WC/runs/gpic-speed LIMIT="" SAMPLERS=(gm_draw gm_sde euler_mean) CALLS=50
while :; do
  case "$1" in
    --root) R=$2; shift 2; [ -d "$R" ] || { echo "no root $R"; exit 2; } ;;
    --limit) LIMIT=$2; shift 2; [[ $LIMIT =~ ^[1-9][0-9]*$ ]] || { echo "--limit takes a batch count"; exit 2; } ;;
    --samplers) read -ra SAMPLERS <<< "$2"; shift 2 ;;
    *) break ;;
  esac
done
[ $# -ge 1 ] && [ ${#SAMPLERS[@]} -ge 1 ] || { echo "usage: evals_gm_ladder.sh [--root <runs root>] [--limit <batches>] [--samplers \"<s> ...\"] <tag>:<d>:<nb>:<nt>:<step> ..."; exit 2; }
for s in "${SAMPLERS[@]}"; do case $s in gm_draw|gm_sde|euler_mean) ;; *) echo "sampler $s is not gm_draw, gm_sde or euler_mean"; exit 2 ;; esac; done
WT=$WC/project/gpic-wt/$WTN
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SHA=$(git -C $WT rev-parse HEAD)
# runs (one per argument) and tasks (one per run and sampler); every argument is checked before any GPU work
RTAG=() RK=() RD=() RNB=() RNT=() RST=() RSRC=() RRAW=()
TRUN=() TSMP=() TOUT=() TNAME=()
for t in "$@"; do
  IFS=: read -r tag d nb nt st extra <<< "$t"
  [[ -z "$extra" && $tag =~ ^gm(8|1)-[0-9a-z.-]+$ ]] || { echo "argument $t is not gm<8|1>-<name>:<d>:<nb>:<nt>:<step>"; exit 2; }
  k=${BASH_REMATCH[1]}
  [[ $d =~ ^[0-9]+$ && $nb =~ ^[0-9]+$ && $nt =~ ^[0-9]+$ && $st =~ ^[0-9]+$ ]] || { echo "argument $t is not gm<8|1>-<name>:<d>:<nb>:<nt>:<step>"; exit 2; }
  s6=$(printf %06d $((10#$st)))
  r=${#RTAG[@]}
  RTAG+=($tag) RK+=($k) RD+=($d) RNB+=($nb) RNT+=($nt) RST+=($((10#$st)))
  RSRC+=($R/gm_ladder/$tag/exp_ladder_d$d/ms-step=$s6.ckpt) RRAW+=($R/raw_view/gm_ladder/$tag/exp_ladder_d$d/ms-step=$s6.ckpt)
  for s in "${SAMPLERS[@]}"; do
    TRUN+=($r) TSMP+=($s) TNAME+=("$tag:$s") TOUT+=($R/raw_view/gm_ladder/$tag/exp_ladder_d$d/eval_${s6}_${s}_calls${CALLS}_official)   # evals_gmflow.sh eval_dir
  done
done
NR=${#RTAG[@]} N=${#TRUN[@]}
MK=$(mktemp -d)   # per run: ok, missing or failed, written by the raw-view loop below
( for r in $(seq 0 $((NR - 1))); do
    if [ -e ${RRAW[$r]} ]; then echo ok > $MK/$r; continue; fi
    if [ ! -f ${RSRC[$r]} ]; then echo missing > $MK/$r; continue; fi
    if RAW_ROOT=$R RAW_SRC=gm_ladder/${RTAG[$r]} $WC/envs/pixgen/bin/python -I $WT/tools/make_raw_view.py d${RD[$r]} ${RST[$r]} > $MK/$r.log 2>&1
    then echo ok > $MK/$r; cat $MK/$r.log; else echo failed > $MK/$r; fi
  done ) &
RVPID=$!
run_mark() {   # <run index>: waits for the run's raw-view mark and prints it; a raw-view loop that died without a mark fails the run instead of waiting forever
  local r=$1
  while [ ! -s $MK/$r ]; do
    kill -0 $RVPID 2>/dev/null || { [ -s $MK/$r ] || echo failed > $MK/$r; break; }
    sleep 2
  done
  cat $MK/$r
}
task_cmd() {   # <task index>: the evaluation's command line (jobs/evals_gmflow.sh, one task); the caller sets EVAL_PHASE or EVAL_LIMIT_BATCHES
  local i=$1 r=${TRUN[$1]}
  bash $WT/jobs/evals_gmflow.sh ${RK[$r]} d${RD[$r]} ${RD[$r]} ${RNB[$r]} ${RNT[$r]} $R/raw_view/gm_ladder/${RTAG[$r]} --wt $WTN ${RST[$r]}:${TSMP[$i]}:$CALLS
}
SKIP=() FAIL=() NEXT=0
chain_next() {   # the next task whose raw view is ready; a missing checkpoint or a failed raw view is recorded and passed over
  local i r
  while [ $NEXT -lt $N ]; do
    i=$NEXT; NEXT=$((NEXT + 1)); r=${TRUN[$i]}
    case $(run_mark $r) in
      missing) echo "task ${TNAME[$i]}: no ${RSRC[$r]}; skipped"; SKIP+=("${TNAME[$i]}"); continue ;;
      failed) echo "task ${TNAME[$i]}: raw view failed:"; cat $MK/$r.log 2>/dev/null; FAIL+=("${TNAME[$i]}"); continue ;;
    esac
    [ "$(git -C $WT rev-parse HEAD)" = "$SHA" ] || { echo "$WT moved from $SHA"; FAIL+=("${TNAME[$i]}"); continue; }
    CHAIN_I=$i; return 0
  done
  CHAIN_I=-1
}
if [ -n "$LIMIT" ]; then   # the smoke test: each task alone, a few batches, the sample check of jobs/eval_gmflow.sh
  while :; do
    chain_next; i=$CHAIN_I; [ $i -ge 0 ] || break
    EVAL_LIMIT_BATCHES=$LIMIT task_cmd $i || FAIL+=("${TNAME[$i]}")
    chk=$(cat ${TOUT[$i]}/sample_check.txt 2>/dev/null | head -c 300)
    echo "task ${TNAME[$i]} sample check: $chk"
    [ -n "$chk" ] || FAIL+=("${TNAME[$i]}")
  done
else
  find /dev/shm -maxdepth 1 -name 'gpic_evals_*' -user "$USER" -mmin +360 -exec rm -rf {} + 2>/dev/null || true   # a killed job's
  export EVAL_SHM_ROOT=$(mktemp -d /dev/shm/gpic_evals_XXXXXX)   # the samples between a task's predict and its scoring
  _prev_exit_cmd=$(trap -p EXIT | sed -E "s/^trap -- '(.*)' EXIT$/\1/; s/'\\\\''/'/g")
  _on_exit() { rm -rf -- "$EVAL_SHM_ROOT" "$MK"; eval "$_prev_exit_cmd"; }
  trap _on_exit EXIT
  chain_cmd() { EVAL_PHASE=$1 task_cmd $2; }   # <phase> <i>
  source $WT/jobs/eval_chain_lib.sh
  chain_run
  for i in $CHAIN_FAILED; do FAIL+=("${TNAME[$i]}"); done
  for i in $(seq 0 $((N - 1))); do   # a task counts as done only with both metric files
    case " ${SKIP[*]} ${FAIL[*]} " in *" ${TNAME[$i]} "*) continue ;; esac
    [ -s ${TOUT[$i]}/metrics.json ] && [ -s ${TOUT[$i]}/metrics_inception.json ] || { echo "task ${TNAME[$i]}: no metrics.json or no metrics_inception.json in ${TOUT[$i]}"; FAIL+=("${TNAME[$i]}"); }
  done
fi
wait $RVPID
for i in $(seq 0 $((N - 1))); do
  res=ok
  for s in "${SKIP[@]}"; do [ "$s" != "${TNAME[$i]}" ] || res=skipped; done
  for f in "${FAIL[@]}"; do [ "$f" != "${TNAME[$i]}" ] || res=failed; done
  echo "task ${TNAME[$i]}: $res"
done
echo "gm ladder evaluations: $N tasks, ${#SKIP[@]} skipped, ${#FAIL[@]} failed"
[ ${#SKIP[@]} -eq 0 ] && [ ${#FAIL[@]} -eq 0 ]
