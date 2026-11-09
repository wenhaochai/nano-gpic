#!/bin/bash
#SBATCH -J gpic-resume-noisecheck
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:55:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The resume cost under each prompt's own noise (gpic-28 2026-11-09). The official evaluation draws ONE noise tensor for all 50K
# prompts, and every FD of the resume isolation used it (seven one-piece runs 802.7-830.0; plain resumes 851.0-884.1; E1 867.0, E2
# 822.8). If the cost shows only with that one tensor, it is a property of the evaluation's noise and not of the weights. This job
# scores finished raw views again with each prompt's own noise (evals_gmflow.sh --noise prompt: Euler 50, CFG 1, the same 50K test
# captions; directory eval_<step>_euler_calls50, the names from before 2026-11-06) as one chain (eval_chain_lib.sh: every predict
# sets up beside the one before it, the scoring runs beside the predicts) and prints both FDs of every run.
# usage: eval_noisecheck.sh <rel>:<size>:<d>:<nb>:<nt>:<step> ...     (runs/gpic-speed/raw_view/<rel> must hold the checkpoint)
#   the planned call (one piece, plain resume, E1, E2):
#     eval_noisecheck.sh muon_fp8recipe/fp8-d20-n3-s2c:d1536:1536:16:4:13600 muon_fp8recipe/fp8-d20-n3-s2r:d1536:1536:16:4:13600 \
#       muon_fp8recipe/fp8-d20-n3-s2rcpu:d1536:1536:16:4:13600 muon_fp8recipe/fp8-d20-n3-s2rgpu:d1536:1536:16:4:13600
# --time: 8.3 min of predict per width-1536 task + the last scoring (~3 min) + the start (~2 min): 4 tasks ~38 min.
export NCCL_NVLS_ENABLE=0   # NCCL 2.26.2 NVLS all-gather can permute rank blocks (NVIDIA/nccl #1906)
set -o pipefail
WC=/scratch/gpfs/GROUP/USER
source $WC/project/gpic/jobs/env.sh
WTN=eval_official_run9   # the code of every FD above
[ $# -ge 1 ] || { echo "usage: eval_noisecheck.sh <rel>:<size>:<d>:<nb>:<nt>:<step> ..."; exit 2; }
WT=$WC/project/gpic-wt/$WTN
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
R=$WC/runs/gpic-speed
TASK=() REL=() SZ=() D=() NB=() NT=() ST=() DIR=()
for t in "$@"; do   # every task is checked before any GPU work
  IFS=: read -r rel sz d nb nt st extra <<< "$t"
  [[ -z "$extra" && $rel =~ ^[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*$ && $sz =~ ^[A-Za-z0-9_-]+$ && $d =~ ^[0-9]+$ && $nb =~ ^[0-9]+$ && $nt =~ ^[0-9]+$ && $st =~ ^[0-9]+$ ]] \
    || { echo "task $t is not <rel>:<size>:<d>:<nb>:<nt>:<step>"; exit 2; }
  s6=$(printf %06d $((10#$st)))
  [ -e $R/raw_view/$rel/exp_ladder_$sz/ms-step=$s6.ckpt ] || { echo "task $t: no raw view $R/raw_view/$rel/exp_ladder_$sz/ms-step=$s6.ckpt"; exit 2; }
  TASK+=("$t") REL+=($rel) SZ+=($sz) D+=($d) NB+=($nb) NT+=($nt) ST+=($((10#$st))) DIR+=($R/raw_view/$rel/exp_ladder_$sz/eval_${s6}_euler_calls50)
done
N=${#TASK[@]}
find /dev/shm -maxdepth 1 -name 'gpic_evals_*' -user "$USER" -mmin +360 -exec rm -rf {} + 2>/dev/null || true   # a killed job's
export EVAL_SHM_ROOT=$(mktemp -d /dev/shm/gpic_evals_XXXXXX)   # the samples between a task's predict and its scoring
_prev_exit_cmd=$(trap -p EXIT | sed -E "s/^trap -- '(.*)' EXIT$/\1/; s/'\\\\''/'/g")
_on_exit() { rm -rf -- "$EVAL_SHM_ROOT"; eval "$_prev_exit_cmd"; }
trap _on_exit EXIT
NEXT=0
chain_next() { if [ $NEXT -lt $N ]; then CHAIN_I=$NEXT; NEXT=$((NEXT + 1)); else CHAIN_I=-1; fi; }
chain_cmd() {   # <phase> <i>
  EVAL_PHASE=$1 bash $WT/jobs/evals_gmflow.sh 0 ${SZ[$2]} ${D[$2]} ${NB[$2]} ${NT[$2]} $R/raw_view/${REL[$2]} --wt $WTN --noise prompt ${ST[$2]}:euler:50
}
source $WT/jobs/eval_chain_lib.sh
chain_run
fail=0
for i in $(seq 0 $((N - 1))); do
  python3 - ${REL[$i]} ${DIR[$i]} <<'PY' || fail=1
import json, os, sys
rel, d = sys.argv[1:]
fd = lambda p: json.load(open(p + "/metrics.json"))["fd"] if os.path.exists(p + "/metrics.json") else None
own, one = fd(d), fd(d + "_official")
print(f"{rel}: FD {own:.2f} with each prompt's own noise" if own is not None else f"{rel}: NO metrics.json in {d}",
      f"; {one:.2f} with the official noise" if one is not None else "", sep="")
sys.exit(own is None)
PY
done
echo "noise check: $N tasks, failed [${CHAIN_FAILED}]"
[ -z "$CHAIN_FAILED" ] && [ $fail -eq 0 ] && echo RESUME_NOISECHECK_DONE
