#!/bin/bash
#SBATCH -J gpic-image-anneal-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of annealing one mixture per image (docs/gmflow.md section 16) through the production path, jobs/train_gmflow.sh, on 8 H100s:
# 1. train_gmflow.sh 8 d640 ... 512 <tag> w1147 --scope image --anneal 196608:768 trains 512 steps (tau 196608 at step 0, geometric down to 1 at step 768), then the
#    same command with 640 steps resumes from last.ckpt to 640, still annealing (review_image_anneal M2). Every printed line (every 50 calls, rank by rank) must be
#    finite and carry the tau of its global step, 196608^(1 - step/768), so the resumed launch must go on from its checkpoint's step with tau > 1. While tau > 1000 the
#    responsibilities must be soft (mean r_neff > 2); nll_exact and mu_spread must be finite. The run records anneal.txt, the saved config holds anneal_tau0 and anneal_steps, and no recompile follows
#    the warm-up (tau is a Python number in the uncompiled loss);
# 2. the same root with another --anneal is refused before any training;
# 3. predictions of the step-512 checkpoint on 512 prompts: gm_draw and euler_mean at 8 calls (EVAL_SCOPE=image), with their sample checks.
# It reports s/step; it fails on a nonzero exit, a non-finite value, a wrong tau, a missing sample check, or a guard that does not refuse. No W&B.
# Code: the frozen worktree named by the first argument (default gpic-wt/gmflow_run7), with no local changes.
# usage: smoke_image_anneal.sh [worktree name under gpic-wt/]
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WTN=${1:-gmflow_run7}
WT=$WC/project/gpic-wt/$WTN
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "anneal_tau0" $WT/baselines/PixelGen/src/diffusion/flow_matching/training_gmflow.py || { echo "$WT holds old code"; exit 2; }
TAG=annealsmoke$(date +%m%d%H%M%S)
OUT=$WC/runs/gpic-speed/gmflow_k8_d640_$TAG   # train_gmflow.sh's root for this tag
mkdir -p $OUT
export WANDB_MODE=disabled TORCH_LOGS=recompiles GPIC_GM_PRINT=50
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
check() {   # <log> <bench dir> <first step of the launch>: finite lines, the tau of each step, soft responsibilities while tau is large, recompiles, rank 0's step time
python3 -I - "$1" "$2" "$3" <<'PY' | tee -a $OUT/phases.log
import json, math, re, statistics, sys
s = open(sys.argv[1], errors="replace").read()
first = int(sys.argv[3])
F = r"([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"   # a number, not \S+: the 8 ranks print at the same time and lines run together (w1168)
rows = re.findall(r"\[gmflow\] call (\d+): loss " + F + " s_u " + F + " n_eff " + F + " max_w " + F + " win_max " + F + r" win_alive (\d+) w_dead " + F
                  + r" step (\d+) tau " + F + " r_neff " + F + " mu_spread " + F + " nll_exact " + F, s)
vals = [(int(r[0]), *map(float, r[1:6]), int(r[6]), float(r[7]), int(r[8]), float(r[9]), float(r[10]), float(r[11]), float(r[12])) for r in rows]
finite = bool(vals) and all(math.isfinite(v) for r in vals for v in r)
want = lambda st: 1.0 if st >= 768 else 196608 ** (1 - st / 768)
bad_tau = [(r[8], r[9]) for r in vals if abs(r[9] / want(r[8]) - 1) > 1e-4]
early = [r[10] for r in vals if r[9] > 1000]
soft = not early or statistics.mean(early) > 2
steps_ok = all(r[8] >= first for r in vals)
after = s.split("[warmup] done")[-1] if "[warmup] done" in s else ""
ok = finite and not bad_tau and soft and steps_ok and "[warmup] done" in s
try:
    st = [r["step_ms"] for r in json.load(open(f"{sys.argv[2]}/steps_rank0.json"))["rows"] if "step_ms" in r][20:]
    step = f"{statistics.median(st) / 1000:.4f} s/step (median of {len(st)} steps after the first 20)"
except Exception as e:
    step = f"no step times ({e!r})"
print("(call, loss, s_u, n_eff, max_w, win_max, win_alive, w_dead, step, tau, r_neff, mu_spread, nll_exact):", vals[:: max(1, len(vals) // 10)])
print(f"anneal check: {'PASS' if ok else 'FAIL'} | lines {len(vals)} finite {finite} | wrong tau {bad_tau[:4]} | mean r_neff while tau > 1000: "
      f"{statistics.mean(early) if early else '-'} | steps from {first}: {steps_ok} | recompiles after warm-up {after.count('__recompiles')} | {step}")
sys.exit(0 if ok and after.count("__recompiles") == 0 else 1)
PY
}
fail=0
ARGS=(--wt $WTN --scope image --anneal 196608:768)
log "START train to 512 ($TAG, K = 8, gm_scope image, anneal 196608:768)"
BENCH_OUT=$OUT/bench/first timeout -k 60 900 bash $WT/jobs/train_gmflow.sh 8 d640 640 9 2 512 $TAG w1147 "${ARGS[@]}" > $OUT/first.out 2>&1
rc=$?; log "END train rc=$rc $(tail -1 $OUT/first.out)"; [ $rc -eq 0 ] || exit $rc
cp $OUT/train.log $OUT/first.log
check $OUT/first.log $OUT/bench/first 0 || fail=1
[ "$(cat $OUT/anneal.txt)" = "196608:768" ] || { log "anneal.txt holds $(cat $OUT/anneal.txt)"; fail=1; }
grep -q "anneal_tau0: 196608" $OUT/exp_ladder_d640/config-*.yaml && grep -q "anneal_steps: 768" $OUT/exp_ladder_d640/config-*.yaml || { log "the saved config lacks the schedule"; fail=1; }
log "START resume to 640"
N=$(wc -c < $OUT/train.log)
BENCH_OUT=$OUT/bench/resume timeout -k 60 600 bash $WT/jobs/train_gmflow.sh 8 d640 640 9 2 640 $TAG w1147 "${ARGS[@]}" > $OUT/resume.out 2>&1
rc=$?; log "END resume rc=$rc $(tail -1 $OUT/resume.out)"; [ $rc -eq 0 ] || fail=1
tail -c +$((N + 1)) $OUT/train.log > $OUT/resume.log
grep -q "resuming from .*last" $OUT/resume.log || { log "the resume did not start from last.ckpt"; fail=1; }
check $OUT/resume.log $OUT/bench/resume 512 || fail=1
timeout -k 30 120 bash $WT/jobs/train_gmflow.sh 8 d640 640 9 2 640 $TAG w1147 --wt $WTN --scope image --anneal 1000:768 > $OUT/guard.out 2>&1
rc=$?
if [ $rc -eq 2 ] && grep -q "started with anneal 196608:768, not 1000:768" $OUT/guard.out; then log "guard: another --anneal refused (rc=$rc)"
else log "guard: another --anneal was not refused with its message (rc=$rc)"; fail=1; fi
export EVAL_LIMIT_BATCHES=2 LADDER_RUN_ROOT=$OUT GMFLOW_WT=$WT
for S in gm_draw euler_mean; do
  log "START predict $S 8"
  EVAL_SCOPE=image timeout -k 30 300 bash $WT/jobs/eval_gmflow.sh 8 512 $S 8 > $OUT/eval_${S}_8.log 2>&1
  rc=$?
  D=$OUT/exp_ladder_d640/eval_000512_${S}_calls8_official
  chk=$(cat $D/sample_check.txt 2>/dev/null)
  log "END predict $S 8 rc=$rc sample_check: $chk"
  [ $rc -eq 0 ] && [ -n "$chk" ] || fail=1
done
log "DONE $OUT fail=$fail"
exit $fail
