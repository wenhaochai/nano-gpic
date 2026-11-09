#!/bin/bash
#SBATCH -J gpic-evals-group
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=03:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# One evaluation job for a group of runs (gpic-28 2026-11-09, for genmodel: the no-EMA sweeps score each run once, at its last step, so
# one job covers several runs, each with its own root, size, d, nb, nt and step). For each task: the raw view of
# runs/gpic-speed/<rel>/exp_ladder_<size>/ms-step=<step>.ckpt (tmp/ladder_prof/make_raw_view.py: the ema slot holds the raw weights), then
# the official evaluation of runs/gpic-speed/raw_view/<rel> at <step>:euler:50 with jobs/evals_gmflow.sh (Euler 50, official noise,
# CFG 1, 50K test captions), so directories and logs are the ones wsd_eval.sh gives. The predicts and the scoring run as a chain
# (jobs/eval_chain_lib.sh, run6: the next predict sets up beside the current one; scoring beside both), and the raw views are written in the background ahead of their predict. A task whose checkpoint is
# missing (a failed training) is skipped and named; the job exits 1 if any task was skipped or failed. Done tasks (metrics.json) are
# skipped, so a rerun does only the rest.
# usage: evals_group.sh [--wt <worktree>] <rel>:<size>:<d>:<nb>:<nt>:<step> ...
#   e.g. evals_group.sh muon_wsd/x-d11-c4-lr6p80e-3:d896:896:11:4:16704 muon_wsd/x-d11-c4-lr1p36e-2:d896:896:11:4:16704
# --time: the sum of the tasks' predict times + 2 min (the scoring hides under the next predict; the last one adds it): predict ~3.3
# min at width 640 (d9), 3.6 at 768, 4.1 at 896, 4.8 at 1024, 6.3 at 1280, 8.3 at 1536, 10.8 at 1792; a raw view ~0.5 min before the first.
export NCCL_NVLS_ENABLE=0   # NCCL 2.26.2 NVLS all-gather can permute rank blocks (NVIDIA/nccl #1906)
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
source $WC/project/gpic/jobs/env.sh
WTN=eval_official_run6
if [ "$1" = "--wt" ]; then WTN=$2; shift 2; [[ $WTN =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--wt takes a worktree name under gpic-wt/"; exit 2; }; fi
[ $# -ge 1 ] || { echo "usage: evals_group.sh [--wt <worktree>] <rel>:<size>:<d>:<nb>:<nt>:<step> ..."; exit 2; }
WT=$WC/project/gpic-wt/$WTN
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SHA=$(git -C $WT rev-parse HEAD)
R=$WC/runs/gpic-speed
TASK=() REL=() SZ=() D=() NB=() NT=() ST=() SRC=() RAW=() OUTD=()
for t in "$@"; do   # every task is checked before any GPU work
  IFS=: read -r rel sz d nb nt st extra <<< "$t"
  [[ -z "$extra" && $rel =~ ^[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*$ && $sz =~ ^[A-Za-z0-9_-]+$ && $d =~ ^[0-9]+$ && $nb =~ ^[0-9]+$ && $nt =~ ^[0-9]+$ && $st =~ ^[0-9]+$ ]] \
    || { echo "task $t is not <rel>:<size>:<d>:<nb>:<nt>:<step>"; exit 2; }
  s6=$(printf %06d $((10#$st)))
  TASK+=("$t") REL+=($rel) SZ+=($sz) D+=($d) NB+=($nb) NT+=($nt) ST+=($((10#$st)))
  SRC+=($R/$rel/exp_ladder_$sz/ms-step=$s6.ckpt) RAW+=($R/raw_view/$rel/exp_ladder_$sz/ms-step=$s6.ckpt)
  OUTD+=($R/raw_view/$rel/exp_ladder_$sz/eval_${s6}_euler_calls50_official)   # evals_gmflow.sh eval_dir with the official noise
done
N=${#TASK[@]}
MK=$(mktemp -d)   # per task: ok, missing or failed, written by the raw-view loop below
( for i in $(seq 0 $((N - 1))); do
    if [ -f ${OUTD[$i]}/metrics.json ] || [ -f ${RAW[$i]} ]; then echo ok > $MK/$i; continue; fi
    if [ ! -f ${SRC[$i]} ]; then echo missing > $MK/$i; continue; fi
    if RAW_SRC=${REL[$i]} $WC/envs/pixgen/bin/python -I $WC/tmp/ladder_prof/make_raw_view.py ${SZ[$i]} ${ST[$i]} > $MK/$i.log 2>&1
    then echo ok > $MK/$i; else echo failed > $MK/$i; fi
  done ) &
RVPID=$!
find /dev/shm -maxdepth 1 -name 'gpic_evals_*' -user "$USER" -mmin +360 -exec rm -rf {} + 2>/dev/null || true   # a killed job's
export EVAL_SHM_ROOT=$(mktemp -d /dev/shm/gpic_evals_XXXXXX)   # the samples between a task's predict and its scoring
_prev_exit_cmd=$(trap -p EXIT | sed -E "s/^trap -- '(.*)' EXIT$/\1/; s/'\\\\''/'/g")
_on_exit() { rm -rf -- "$EVAL_SHM_ROOT" "$MK"; eval "$_prev_exit_cmd"; }
trap _on_exit EXIT
SKIP=() FAIL=() NEXT=0
chain_next() {   # the next task whose raw view is ready; a missing checkpoint or a failed raw view is recorded and passed over
  local i t
  while [ $NEXT -lt $N ]; do
    i=$NEXT; NEXT=$((NEXT + 1)); t=${TASK[$i]}
    while [ ! -s $MK/$i ]; do   # a raw-view loop that died (killed) without a mark fails the task instead of waiting forever
      kill -0 $RVPID 2>/dev/null || { [ -s $MK/$i ] || echo failed > $MK/$i; break; }
      sleep 2
    done
    case $(cat $MK/$i) in
      missing) echo "task $t: no ${SRC[$i]}; skipped"; SKIP+=("$t"); continue ;;
      failed) echo "task $t: raw view failed:"; cat $MK/$i.log 2>/dev/null; FAIL+=("$t"); continue ;;
    esac
    [ "$(git -C $WT rev-parse HEAD)" = "$SHA" ] || { echo "$WT moved from $SHA"; FAIL+=("$t"); continue; }
    CHAIN_I=$i; return 0
  done
  CHAIN_I=-1
}
chain_cmd() {   # <phase> <i>
  EVAL_PHASE=$1 bash $WT/jobs/evals_gmflow.sh 0 ${SZ[$2]} ${D[$2]} ${NB[$2]} ${NT[$2]} $R/raw_view/${REL[$2]} --wt $WTN ${ST[$2]}:euler:50
}
source $WT/jobs/eval_chain_lib.sh
chain_run
for i in $CHAIN_FAILED; do FAIL+=("${TASK[$i]}"); done
wait $RVPID
for t in "${TASK[@]}"; do
  r=ok; for s in "${SKIP[@]}"; do [ "$s" != "$t" ] || r=skipped; done; for f in "${FAIL[@]}"; do [ "$f" != "$t" ] || r=failed; done
  echo "task $t: $r"
done
echo "group: $N tasks, ${#SKIP[@]} skipped, ${#FAIL[@]} failed"
[ ${#SKIP[@]} -eq 0 ] && [ ${#FAIL[@]} -eq 0 ]
