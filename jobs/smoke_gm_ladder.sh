#!/bin/bash
#SBATCH -J gpic-gm-ladder-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of the per-pixel Gaussian mixture on the final ladder's trainer (phase 2; owner popups 2026-11-12 22:5x: six trainings approved, to be
# submitted after this test), on 8 H100s, with the production scripts of the frozen worktree gpic-wt/gmflow_ladder_run in a root of this smoke:
#   A   jobs/train_gmflow_ladder.sh, gm8 at d10 (768 x (10+2), DDP): a whole run of 320 steps (warmup 3, cooldown over the last 160), stopped at step 160
#       by --trainer.max_steps;
#   A2  the same command without the stop: it must resume from last.ckpt at step 160 and end at step 320 with ms-step=000320.ckpt;
#   C   gm8 at d14 (1280 x (14+3), FSDP2 group 7): 320 steps in one piece;
#   B   gm1 at d10: 320 steps in one piece;
#   E1  jobs/evals_gm_ladder.sh --limit 2 (2 batches per rank, the sample check, no metrics): gm_draw on B's and C's checkpoints;
#   H   jobs/heldout_gm_ladder.sh: the held-out loss of the mixture mean and the NLL on the three final checkpoints (3072 + 3072 images);
#   E2  jobs/evals_gm_ladder.sh in full on A2's checkpoint: gm_draw, gm_sde and euler_mean at 50 calls on 50K captions, chained, with both metric files;
#   P   the evaluation code against the ladder's: the ladder's d10 point run xl-d10-0p25c0 (step 21248) evaluated again here with Euler at 50 calls; its
#       DINOv2 FD must be within 1% of the ladder's record (949.77, code eval_official_run11), so the mixture's and the point model's numbers compare.
# Each run lives in a directory of this smoke (SWEEP_RUN_ROOT, --root), with no W&B. It fails on a nonzero exit, a non-finite loss, step files of other than
# 8 ranks, a peak above 74 of the 79.6 GiB an H100 reports, a GPU time per step over 1.3 times the point run's step time times 1 + r (measured on the
# ladder's runs: 0.0762 s at d10, 0.2017 s at d14), an NCCL other than 2.27.7, the wrong strategy, no '[warmup] done', a missing checkpoint, a raw view that
# is not a link (the EMA slot must hold the raw weights), a missing sample check, metric file or held-out record, or a run directory that appeared under
# runs/gpic-speed/gm_ladder/. usage: smoke_gm_ladder.sh
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/gmflow_ladder_run
TR=$WT/jobs/train_gmflow_ladder.sh
[ -f $TR ] && [ -f $WT/jobs/evals_gm_ladder.sh ] && [ -f $WT/jobs/heldout_gm_ladder.sh ] || { echo "$WT lacks the mixture's job scripts"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/gm_ladder_smoke_$(date +%m%d_%H%M%S)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
check() {   # <arm> <log> <bench dir> <estimated s/step> <FSDP2 expected: 0|1> <first step expected> <last step expected>
python3 -I - "$@" <<'PY' | tee -a $OUT/phases.log
import glob, json, math, re, statistics, sys
arm, logf, bench, est, fsdp, first, last = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5] == "1", int(sys.argv[6]), int(sys.argv[7])
s = open(logf, errors="replace").read()
bad, gp, st, losses, r0, peak, nrank, head = [], [], [], [], [], 0.0, 0, {}
for f in sorted(glob.glob(f"{bench}/steps_rank*.json")):
    j = json.load(open(f)); rows = j["rows"]; nrank += 1
    peak = max([peak] + [r["max_reserved_gb"] for r in rows])
    losses += [r["loss"] for r in rows if "loss" in r]
    if f.endswith("steps_rank0.json"):
        gp = [r["gpu_ms"] for r in rows if "gpu_ms" in r][20:]   # first to last kernel of a step: no wait for data
        st = [r["step_ms"] for r in rows if "step_ms" in r][20:]
        r0 = [(r["step"], round(r["loss"], 4)) for r in rows if "loss" in r]
        head = {k: round(j[k], 1) for k in ("process_start_to_fit_start_s", "fit_start_to_train_start_s", "train_start_to_first_step_end_s") if k in j}
if nrank != 8: bad.append(f"step files of {nrank} ranks, not 8")
if not losses or not all(math.isfinite(v) for v in losses): bad.append("no loss or a non-finite loss")
if not 1 < peak <= 74.0: bad.append(f"peak {peak:.1f} GiB outside (1, 74]")
gmed = statistics.median(gp) / 1000 if gp else float("nan")
smed = statistics.median(st) / 1000 if st else float("nan")
if not gmed <= 1.3 * est: bad.append(f"{gmed:.4f} s of GPU time per step is over 1.3 x the estimate {est}")
nccl = re.findall(r"NCCL version ([0-9][0-9.]*)", s)
if not nccl or any(v != "2.27.7" for v in nccl): bad.append(f"NCCL versions {sorted(set(nccl))}")
n_fsdp = s.count("FSDP2: blocks sharded")
if (n_fsdp > 0) != fsdp: bad.append(f"{n_fsdp} FSDP2 lines, FSDP2 expected: {fsdp}")
if "[warmup] done" not in s: bad.append("no '[warmup] done'")
steps = [x for x, _ in r0]
if not steps or min(steps) > first + 2 or max(steps) < last - 2: bad.append(f"rank 0 steps {steps[:1]}..{steps[-1:]}, expected {first}..{last}")
after = s.split("[warmup] done")[-1] if "[warmup] done" in s else ""
print(f"gm smoke {arm}:", "PASS" if not bad else "FAIL " + "; ".join(bad))
print(f"  {arm}: GPU time {gmed:.4f} s/step (estimate {est}, ratio {gmed / est:.3f}), start to start {smed:.4f} s/step; peak reserved {peak:.1f} GiB; NCCL {sorted(set(nccl))}; "
      f"FSDP2 lines {n_fsdp}; recompiles after the warm-up {after.count('__recompiles')}; start-up (s) {head}")
print(f"  {arm}: loss on rank 0 (step, loss): first {r0[:2]}, middle {r0[len(r0) // 2: len(r0) // 2 + 2]}, last {r0[-2:]}")
sys.exit(1 if bad else 0)
PY
}
fail=0
train() {   # <arm> <inner limit s> <estimated s/step> <FSDP2: 0|1> <first step> <last step> <the 11 list columns> [main.py args]
  local n=$1 lim=$2 est=$3 fs=$4 s0=$5 s1=$6; shift 6
  local prod=$WC/runs/gpic-speed/gm_ladder/$1 before=0
  [ -e $prod ] && before=1
  log "START $n: $*"
  SWEEP_RUN_ROOT=$OUT/gm_ladder/$1 BENCH_OUT=$OUT/bench/$n timeout -k 60 $lim bash $TR "$@" > $OUT/$n.log 2>&1
  local rc=$?
  log "END $n rc=$rc"
  [ $rc -eq 0 ] || { fail=1; tail -n 40 $OUT/$n.log; }
  check $n $OUT/$n.log $OUT/bench/$n $est $fs $s0 $s1 || fail=1
  if [ $before -eq 0 ] && [ -e $prod ]; then log "FAIL: the smoke created $prod"; fail=1; fi
}
A="gm8-d10-smoke 768 10 2 320 1.360e-02 0.000501 0.01824 full 1024 50"
train A  900 0.0803 0 0 160 $A --trainer.max_steps 160
[ -f $OUT/gm_ladder/gm8-d10-smoke/exp_ladder_d768/last.ckpt ] || { log "FAIL: A left no last.ckpt"; fail=1; }
train A2 900 0.0803 0 160 320 $A
grep -q "resuming from .*last" $OUT/A2.log || { log "FAIL: A2's log shows no resume"; fail=1; }
train C 1500 0.2064 1 0 320 gm8-d14-smoke 1280 14 3 320 3.400e-03 0.000238 0.0168 full 1024 50
train B  900 0.0763 0 0 320 gm1-d10-smoke 768 10 2 320 1.360e-02 0.000501 0.01735 full 1024 50
for r in "gm8-d10-smoke d768" "gm1-d10-smoke d768" "gm8-d14-smoke d1280"; do
  set -- $r
  [ -f $OUT/gm_ladder/$1/exp_ladder_$2/ms-step=000320.ckpt ] || { log "FAIL: no gm_ladder/$1/exp_ladder_$2/ms-step=000320.ckpt"; fail=1; }
done
# E1: the evaluation of the K = 1 and the FSDP2 checkpoints, 2 batches per rank
log "START E1 (gm_draw, 2 batches per rank: gm1 d10, gm8 d14)"
bash $WT/jobs/evals_gm_ladder.sh --root $OUT --limit 2 --samplers "gm_draw" gm1-d10-smoke:768:10:2:320 gm8-d14-smoke:1280:14:3:320 > $OUT/E1.log 2>&1
rc=$?
grep "sample check\|^task \|^gm ladder evaluations" $OUT/E1.log | cut -c1-400 | tee -a $OUT/phases.log
log "END E1 rc=$rc"
[ $rc -eq 0 ] || { fail=1; tail -n 30 $OUT/E1.log; tail -n 30 $OUT/raw_view/gm_ladder/*/eval_*.log 2>/dev/null; }
for r in "gm1-d10-smoke d768" "gm8-d14-smoke d1280"; do
  set -- $r
  [ -L $OUT/raw_view/gm_ladder/$1/exp_ladder_$2/ms-step=000320.ckpt ] || { log "FAIL: the raw view of $1 is not a link: its EMA slot does not hold the raw weights"; fail=1; }
done
# H: the held-out loss of the three final checkpoints
log "START H (held-out loss and NLL of 3 checkpoints)"
bash $WT/jobs/heldout_gm_ladder.sh --root $OUT --out $OUT/heldout.jsonl > $OUT/H.log 2>&1
rc=$?
log "END H rc=$rc"
[ $rc -eq 0 ] || { fail=1; tail -n 30 $OUT/H.log; }
python3 -I - $OUT/heldout.jsonl <<'PY' | tee -a $OUT/phases.log || fail=1
import json, math, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1])]
except OSError as e:
    print("gm smoke H: FAIL", e); sys.exit(1)
fin = lambda r, k: isinstance(r.get(k), float) and math.isfinite(r[k])
runs = {r["ckpt"].split("/")[-3] for r in rows}
ok = runs == {"gm8-d10-smoke", "gm1-d10-smoke", "gm8-d14-smoke"} and all(
    fin(r, "loss_val") and fin(r, "loss_train") and fin(r, "nll_val") and fin(r, "gauss_nll_val") and r.get("gm_k") == int(r["ckpt"].split("/")[-3][2]) for r in rows)
for r in rows:
    print(f"  held-out {r['ckpt'].split('/')[-3]} {r.get('weights')}: gm_k {r.get('gm_k')}, loss_val {r.get('loss_val')}, loss_train {r.get('loss_train')}, "
          f"nll_val {r.get('nll_val')}, gauss_nll_val {r.get('gauss_nll_val')}, {r.get('elapsed_s')} s")
print("gm smoke H:", "PASS" if ok else f"FAIL (runs {sorted(runs)})")
sys.exit(0 if ok else 1)
PY
# E2: the three samplers in full on A2's checkpoint, chained
log "START E2 (gm8 d10: gm_draw, gm_sde, euler_mean at 50 calls, 50K captions)"
bash $WT/jobs/evals_gm_ladder.sh --root $OUT gm8-d10-smoke:768:10:2:320 > $OUT/E2.log 2>&1
rc=$?
grep "^task \|^gm ladder evaluations" $OUT/E2.log | cut -c1-300 | tee -a $OUT/phases.log
log "END E2 rc=$rc"
[ $rc -eq 0 ] || { fail=1; tail -n 30 $OUT/E2.log; tail -n 20 $OUT/raw_view/gm_ladder/gm8-d10-smoke/eval_*.log 2>/dev/null; }
[ -L $OUT/raw_view/gm_ladder/gm8-d10-smoke/exp_ladder_d768/ms-step=000320.ckpt ] || { log "FAIL: the raw view of gm8-d10-smoke is not a link"; fail=1; }
# P: the ladder's d10 point run again with this code
log "START P (xl-d10-0p25c0 step 21248, Euler at 50 calls, with this code)"
bash $WT/jobs/evals_gmflow.sh 0 d768 768 10 2 $OUT/point_check --link $WC/runs/gpic-speed/muon_sweep/xl-d10-0p25c0/exp_ladder_d768 --wt gmflow_ladder_run 21248:euler:50 > $OUT/P.log 2>&1
rc=$?
log "END P rc=$rc"
[ $rc -eq 0 ] || { fail=1; tail -n 30 $OUT/P.log; tail -n 20 $OUT/point_check/eval_*.log 2>/dev/null; }
python3 -I - $OUT $WC/runs/gpic-speed/raw_view/muon_sweep/xl-d10-0p25c0/exp_ladder_d768/eval_021248_euler_calls50_official <<'PY' | tee -a $OUT/phases.log || fail=1
import glob, json, math, os, sys
out, ref = sys.argv[1], sys.argv[2]
ok = True
def load(d):
    m = json.load(open(os.path.join(d, "metrics.json")))
    m.update(json.load(open(os.path.join(d, "metrics_inception.json"))))
    return m
for s in ("gm_draw", "gm_sde", "euler_mean"):
    d = f"{out}/raw_view/gm_ladder/gm8-d10-smoke/exp_ladder_d768/eval_000320_{s}_calls50_official"
    try:
        m = load(d)
        good = m.get("n") == 50000 and all(math.isfinite(m[k]) for k in ("fd", "inception_fd", "precision", "recall", "mmd"))
        print(f"  E2 {s}: n {m.get('n')}, DINOv2 FD {m['fd']:.1f}, Inception FD {m['inception_fd']:.2f}, precision {m['precision']:.4f}, recall {m['recall']:.4f}; times {json.load(open(os.path.join(d, 'phase_times.json')))}")
    except (OSError, KeyError, ValueError) as e:
        good = False; print(f"  E2 {s}: no metrics ({e})")
    ok = ok and good
print("gm smoke E2:", "PASS" if ok else "FAIL")
try:
    a, b = load(f"{out}/point_check/exp_ladder_d768/eval_021248_euler_calls50_official"), load(ref)
    rel = {k: abs(a[k] - b[k]) / abs(b[k]) for k in ("fd", "inception_fd")}
    same = all(v < 0.01 for v in rel.values())
    print(f"  P: DINOv2 FD {a['fd']:.3f} against the ladder's {b['fd']:.3f} (relative difference {rel['fd']:.5f}); Inception FD {a['inception_fd']:.3f} against {b['inception_fd']:.3f} ({rel['inception_fd']:.5f})")
except (OSError, KeyError, ValueError) as e:
    same = False; print(f"  P: no metrics ({e})")
print("gm smoke P:", "PASS" if same else "FAIL")
sys.exit(0 if ok and same else 1)
PY
log "DONE $OUT fail=$fail"
exit $fail
