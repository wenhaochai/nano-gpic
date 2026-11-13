#!/bin/bash
#SBATCH -J gpic-full-mfu-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Gating smoke of the per-width code (docs/ladder_plan.md section 12.7) before d1280, d1536 and d1792 start on the MFU worktree, on 8 H100s:
# 1. d1792 through this commit's jobs/train_full.sh, the batch path of w1330-w1332 (FULL_WT = the worktree named by the argument, FULL_ROOT = a
#    scratch root, W&B off), with C0 = 800 steps: segment 1 trains 0 -> 800 (ms checkpoints at 200, 400, 600, 800; last.ckpt every 100) and
#    segment 2 (multiple 2) resumes from last.ckpt at 800 and trains to 1600. Each must exit 0 with its final ms checkpoint and last.ckpt
#    complete (torch.load: the step, the denoiser and the EMA), with the MFU commit as the run's first launch and fsdp=1 in its levers line
#    (the final writes of an FSDP2 run were lost before f6713be);
# 2. the full ladder's evaluation path (gmflow_run4's eval_gmflow.sh, K = 0, euler at 8 calls, 1 batch per rank) on segment 2's ms-step=001600,
#    and on step 4800 of the d640 smokes of FSDP2 (full_smoke_1106_1633, ab44db7) and DDP (full_smoke_1106_1433, 5cd3372) side by side.
#    An EMA that still holds most of its initial weights gives flat samples, and the sample check then exits 1, so the gate is that the
#    checkpoint loads with the right shapes and the samples are written (sample_check.txt).
# Exit 1 when any part fails. usage: smoke_full_mfu.sh <worktree of this commit, a name under gpic-wt/>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/${1:?usage: smoke_full_mfu.sh <worktree name>}
MFU_WT=$WC/project/gpic-wt/ladder_full_mfu
EV_WT=$WC/project/gpic-wt/gmflow_run4
for w in $WT $MFU_WT $EV_WT; do
  [ -e "$w/.git" ] || { echo "no worktree $w"; exit 2; }
  [ -z "$(git -C $w status --porcelain --untracked-files=no)" ] || { echo "$w has local changes"; exit 2; }
done
grep -q "ladder_full_mfu" $WT/jobs/train_ladder.sh || { echo "$WT has no per-width code"; exit 2; }
MFU_SHA=$(git -C $MFU_WT rev-parse HEAD)
SMK=$WC/runs/gpic-speed/full_mfu_smoke_$(date +%m%d_%H%M)
mkdir -p $SMK
echo "$(git -C $WT rev-parse HEAD) mfu $MFU_SHA" > $SMK/commit.txt
export WANDB_MODE=disabled FULL_WT=$WT FULL_ROOT=$SMK
fail=0
log() { echo "[$(date +%T)] $*" | tee -a $SMK/phases.log; }
ckcheck() {   # <checkpoint> <step>: it loads, holds that step, and holds the denoiser and the EMA with the same tensors
  python - "$1" "$2" <<'PY' 2>&1 | tee -a $SMK/phases.log
import sys, torch
p, step = sys.argv[1], int(sys.argv[2])
try:
    ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["state_dict"]
    den = {k[len("denoiser."):]: v.shape for k, v in sd.items() if k.startswith("denoiser.")}
    ema = {k[len("ema_denoiser."):]: v.shape for k, v in sd.items() if k.startswith("ema_denoiser.")}
    ok = ck.get("global_step") == step and len(den) > 0 and den == ema
    print(f"{p}: global_step {ck.get('global_step')} (want {step}), {len(den)} denoiser and {len(ema)} EMA tensors, same shapes {den == ema},"
          f" optimizer states {len(ck.get('optimizer_states', []))}: {'ok' if ok else 'FAIL'}")
except Exception as e:
    ok = False
    print(f"{p}: FAIL {e!r}")
sys.exit(0 if ok else 1)
PY
}
R=$SMK/exp_ladder_d1792
for seg in 1 2; do
  log "START d1792 segment $seg (to $((800 * seg)) steps)"
  bash $WT/jobs/train_full.sh d1792 1792 18 4 800 $seg > $SMK/seg$seg.log 2>&1
  rc=$?
  log "END d1792 segment $seg rc=$rc"
  [ $rc -eq 0 ] || fail=1
  S6=$(printf %06d $((800 * seg)))
  ckcheck $R/ms-step=$S6.ckpt $((800 * seg)) || fail=1
  ckcheck "$(ls -t $R/last*.ckpt 2>/dev/null | head -1)" $((800 * seg)) || fail=1
done
grep -q "resuming from .*last" $SMK/seg2.log || { log "segment 2 did not resume from last.ckpt"; fail=1; }
FIRST=$(awk 'NR == 1 {print $2; exit}' $R/launches.txt 2>/dev/null)
[ "$FIRST" = "$MFU_SHA" ] || { log "the run's first launch is $FIRST, not the MFU commit $MFU_SHA"; fail=1; }
grep -q "fsdp=1" $R/launches.txt || { log "no fsdp=1 in launches.txt"; fail=1; }
grep -E "MFU levers|shard log kept|$MFU_SHA" $R/launches.txt | cut -c1-200 | tee -a $SMK/phases.log
grep -h "d1792: the MFU worktree" $SMK/seg1.log $SMK/seg2.log | tee -a $SMK/phases.log
ev() {   # <tag> <run root> <size> <d> <nb> <nt> <step>
  log "START eval $1"
  LADDER_RUN_ROOT=$2 EVAL_SIZE=$3 EVAL_D=$4 EVAL_NB=$5 EVAL_NT=$6 EVAL_LIMIT_BATCHES=1 GMFLOW_WT=$EV_WT \
    timeout -k 30 600 bash $EV_WT/jobs/eval_gmflow.sh 0 $7 euler 8 > $SMK/eval_$1.log 2>&1
  local rc=$?
  local chk=$(cat $2/exp_ladder_$3/eval_$(printf %06d $7)_euler_calls8_official/sample_check.txt 2>/dev/null)
  log "END eval $1 rc=$rc sample_check: $chk"
  [ -n "$chk" ] || fail=1
}
ev d1792 $SMK d1792 1792 18 4 1600
for s in full_smoke_1106_1633 full_smoke_1106_1433; do
  mkdir -p $SMK/$s/exp_ladder_d640
  ln -s $WC/runs/gpic-speed/$s/resume/exp_ladder_d640/ms-step=004800.ckpt $SMK/$s/exp_ladder_d640/ || fail=1
  ev $s $SMK/$s d640 640 9 2 4800
done
log "DONE fail=$fail $SMK"
exit $fail
