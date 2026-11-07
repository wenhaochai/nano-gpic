#!/bin/bash
#SBATCH -J gpic-full-gmflow-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke of GMFlow on the full ladder (docs/gmflow.md section 17) through the production path on 8 H100s, GPIC Full (8000 shards, the full text cache):
# 1. d640, K = 8, through jobs/train_full_gmflow.sh of the worktree named by the argument (FULL_WT; FULL_ROOT = a scratch root; W&B off) with C0 = 800 steps:
#    segment 1 trains 0 -> 800 (ms checkpoints at 200 ... 800; last.ckpt every 100), segment 2 (multiple 2) resumes from last.ckpt at 800 and trains to 1600.
#    Each must exit 0 with its final ms checkpoint and last.ckpt complete (the step, the denoiser and the EMA with the same shapes, the GMFlow head of
#    K x 1024 rows); the run's first launch is the worktree's commit; segment 2 resumes from last.ckpt and keeps the shard log up to its checkpoint; the saved
#    config holds gm_k 8, GMFlowTrainer with lam 0.5, width 640 and the 8000-shard pattern; the [gmflow] lines (every 100 calls, rank by rank) are finite;
# 2. the refusals of train_full_gmflow.sh: d1280, C0 steps not a multiple of 800, K = 0;
# 3. the evaluation path of the full ladder's GMFlow jobs (gpic-wt/gmflow_run4's eval_gmflow.sh, K = 8, 1 batch per rank) on ms-step=001600: GM-ODE and Euler on
#    the mean at 8 calls. The EMA of a 1600-step run still holds most of its initial weights, so a flat sample can fail the sample check; the gate is that the
#    checkpoint loads and the samples are written (sample_check.txt).
# Exit 1 when any part fails. usage: smoke_full_gmflow.sh <worktree of this commit, a name under gpic-wt/>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/${1:?usage: smoke_full_gmflow.sh <worktree name>}
EV_WT=$WC/project/gpic-wt/gmflow_run4
for w in $WT $EV_WT; do
  [ -e "$w/.git" ] || { echo "no worktree $w"; exit 2; }
  [ -z "$(git -C $w status --porcelain --untracked-files=no)" ] || { echo "$w has local changes"; exit 2; }
done
[ -f $WT/jobs/train_full_gmflow.sh ] || { echo "$WT has no train_full_gmflow.sh"; exit 2; }
SHA=$(git -C $WT rev-parse HEAD)
SMK=$WC/runs/gpic-speed/full_gmflow_smoke_$(date +%m%d_%H%M)
mkdir -p $SMK
echo "$SHA" > $SMK/commit.txt
export WANDB_MODE=disabled FULL_WT=$WT FULL_ROOT=$SMK GPIC_GM_PRINT=100 TORCH_LOGS=recompiles
fail=0
log() { echo "[$(date +%T)] $*" | tee -a $SMK/phases.log; }
ckcheck() {   # <checkpoint> <step>: it loads, holds that step, the denoiser and the EMA with the same shapes, and the GMFlow head of K x 1024 rows
  python - "$1" "$2" <<'PY' 2>&1 | tee -a $SMK/phases.log
import sys, torch
p, step = sys.argv[1], int(sys.argv[2])
try:
    ck = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
    sd = ck["state_dict"]
    den = {k[len("denoiser."):]: v.shape for k, v in sd.items() if k.startswith("denoiser.")}
    ema = {k[len("ema_denoiser."):]: v.shape for k, v in sd.items() if k.startswith("ema_denoiser.")}
    head = den.get("final_layer.linear.weight")
    ok = ck.get("global_step") == step and len(den) > 0 and den == ema and head is not None and head[0] == 8 * 1024
    print(f"{p}: global_step {ck.get('global_step')} (want {step}), {len(den)} denoiser and {len(ema)} EMA tensors, same shapes {den == ema},"
          f" head {tuple(head) if head is not None else None}, optimizer states {len(ck.get('optimizer_states', []))}: {'ok' if ok else 'FAIL'}")
except Exception as e:
    ok = False
    print(f"{p}: FAIL {e!r}")
sys.exit(0 if ok else 1)
PY
}
R=$SMK/exp_ladder_d640
for seg in 1 2; do
  log "START d640 GMFlow K = 8 segment $seg (to $((800 * seg)) steps)"
  bash $WT/jobs/train_full_gmflow.sh 8 d640 640 9 2 800 $seg > $SMK/seg$seg.log 2>&1
  rc=$?
  log "END d640 segment $seg rc=$rc"
  [ $rc -eq 0 ] || fail=1
  ckcheck $R/ms-step=$(printf %06d $((800 * seg))).ckpt $((800 * seg)) || fail=1
  ckcheck "$(ls -t $R/last*.ckpt 2>/dev/null | head -1)" $((800 * seg)) || fail=1
done
grep -q "resuming from .*last" $SMK/seg2.log || { log "segment 2 did not resume from last.ckpt"; fail=1; }
FIRST=$(awk 'NR == 1 {print $2; exit}' $R/launches.txt 2>/dev/null)
[ "$FIRST" = "$SHA" ] || { log "the run's first launch is $FIRST, not $SHA"; fail=1; }
grep -q "shard log kept up to .*(checkpoint step 800)" $R/launches.txt || { log "segment 2 kept no shard log up to its checkpoint"; fail=1; }
grep -E "shard log kept|max_steps" $R/launches.txt | cut -c1-200 | tee -a $SMK/phases.log
CFG=$(ls -t $R/config-*.yaml 2>/dev/null | head -1)
for pat in "gm_k: 8" "GMFlowTrainer" "lam: 0.5" "hidden_size: 640" "gpic_train_{00000..07999}.tar"; do
  grep -qF "$pat" $CFG || { log "the saved config $CFG lacks $pat"; fail=1; }
done
python3 -I - $SMK/seg1.log $SMK/seg2.log <<'PY' 2>&1 | tee -a $SMK/phases.log || fail=1
import math, re, sys
F = r"([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"
rows = [r for f in sys.argv[1:] for r in re.findall(r"\[gmflow\] call (\d+): loss " + F + " s_u " + F + " n_eff " + F + " max_w " + F, open(f, errors="replace").read())]
ok = len(rows) >= 8 and all(math.isfinite(float(v)) for r in rows for v in r[1:])
after = "".join(open(f, errors="replace").read().split("[warmup] done")[-1] for f in sys.argv[1:])
print(f"[gmflow] lines {len(rows)}, all finite {ok}; first {rows[:1]}, last {rows[-1:]}; recompiles after warm-up {after.count('__recompiles')}")
sys.exit(0 if ok else 1)
PY
# 2. refusals
for a in "8 d1280 1280 14 3 800 1|runs d640 to d1024" "8 d640 640 9 2 1000 1|multiple of 800" "0 d640 640 9 2 800 1|K must be"; do
  args=${a%%|*} msg=${a##*|}
  out=$(bash $WT/jobs/train_full_gmflow.sh $args 2>&1); rc=$?
  if [ $rc -eq 2 ] && grep -q "$msg" <<< "$out"; then log "refused: $args (rc 2)"; else log "not refused with its message: $args (rc $rc: $out)"; fail=1; fi
done
# 3. the evaluation path
for S in gm_ode euler_mean; do
  log "START eval $S 8"
  LADDER_RUN_ROOT=$SMK EVAL_LIMIT_BATCHES=1 GMFLOW_WT=$EV_WT timeout -k 30 600 bash $EV_WT/jobs/eval_gmflow.sh 8 1600 $S 8 > $SMK/eval_$S.log 2>&1
  rc=$?
  chk=$(cat $R/eval_001600_${S}_calls8_official/sample_check.txt 2>/dev/null)
  log "END eval $S 8 rc=$rc sample_check: $chk"
  [ -n "$chk" ] || fail=1
done
log "DONE fail=$fail $SMK"
exit $fail
