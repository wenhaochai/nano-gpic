#!/bin/bash
#SBATCH -J gpic-ddm-ladder-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of the distributional denoiser on the final ladder's trainer (docs/ddm.md; owner popup 2026-11-22 via the controller, option A: P1 at d10
# first, smoke at prio 20), on 8 H100s, with the production scripts of the frozen worktree gpic-wt/ddm_run in a root of this smoke:
#   A   jobs/train_ddm_ladder.sh at d10 (768 x (10+2), DDP): a whole run of 320 steps (warmup 3, cooldown over the last 160), stopped at step 160 by
#       --trainer.max_steps;
#   A2  the same command without the stop: it must resume from last.ckpt at step 160 and end at step 320 with ms-step=000320.ckpt;
#   E1  jobs/evals_ddm_ladder.sh --limit 2 (2 batches per rank, the sample check, no metrics) at 4 calls;
#   H   jobs/heldout_ddm_ladder.sh: the ddm scores of A2's checkpoint (4 draws: the loss of the draws' mean, the energy score per token) and the energy
#       score of the ladder's d10 point run (3072 + 3072 images);
#   E2  jobs/evals_ddm_ladder.sh in full at 4 calls: 50K captions, both metric files.
# It fails on a nonzero exit, a non-finite loss, step files of other than 8 ranks, a peak above 74 of the 79.6 GiB an H100 reports, a GPU time per step
# over 1.3 times the point run's step time times 1 + r (0.0762 s at d10, r = 0.8237: tools/ddm_ladder/steps.py), an NCCL other than 2.27.7, FSDP2, no
# '[warmup] done', a missing checkpoint, a raw view that is not a link (the EMA slot must hold the raw weights), a missing sample check, metric file or
# held-out record, a ddm held-out loss that is not finite, or a run directory that appeared under runs/gpic-speed/ddm_ladder/. usage: smoke_ddm_ladder.sh
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ddm_run
TR=$WT/jobs/train_ddm_ladder.sh
[ -f $TR ] && [ -f $WT/jobs/evals_ddm_ladder.sh ] && [ -f $WT/jobs/heldout_ddm_ladder.sh ] || { echo "$WT lacks the ddm job scripts"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/ddm_ladder_smoke_$(date +%m%d_%H%M%S)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles GPIC_IMPLICIT_PRINT=40
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
print(f"ddm smoke {arm}:", "PASS" if not bad else "FAIL " + "; ".join(bad))
print(f"  {arm}: GPU time {gmed:.4f} s/step (estimate {est}, ratio {gmed / est:.3f}), start to start {smed:.4f} s/step; peak reserved {peak:.1f} GiB; NCCL {sorted(set(nccl))}; "
      f"FSDP2 lines {n_fsdp}; recompiles after the warm-up {after.count('__recompiles')}; start-up (s) {head}")
print(f"  {arm}: loss on rank 0 (step, loss): first {r0[:2]}, middle {r0[len(r0) // 2: len(r0) // 2 + 2]}, last {r0[-2:]}")
sys.exit(1 if bad else 0)
PY
}
fail=0
train() {   # <arm> <inner limit s> <estimated s/step> <FSDP2: 0|1> <first step> <last step> <the 11 list columns> [main.py args]
  local n=$1 lim=$2 est=$3 fs=$4 s0=$5 s1=$6; shift 6
  local prod=$WC/runs/gpic-speed/ddm_ladder/$1 before=0
  [ -e $prod ] && before=1
  log "START $n: $*"
  SWEEP_RUN_ROOT=$OUT/ddm_ladder/$1 BENCH_OUT=$OUT/bench/$n timeout -k 60 $lim bash $TR "$@" > $OUT/$n.log 2>&1
  local rc=$?
  log "END $n rc=$rc"
  [ $rc -eq 0 ] || { fail=1; tail -n 40 $OUT/$n.log; }
  check $n $OUT/$n.log $OUT/bench/$n $est $fs $s0 $s1 || fail=1
  grep "^\[implicit\]" $OUT/$n.log | tail -n 4 | tee -a $OUT/phases.log
  if [ $before -eq 0 ] && [ -e $prod ]; then log "FAIL: the smoke created $prod"; fail=1; fi
}
A="ddm-d10-smoke 768 10 2 320 1.360e-02 0.000501 0.03156 full 1024 50"   # the production run's Muon decay (5 / (LR x 11648))
train A  900 0.1397 0 0 160 $A --trainer.max_steps 160
[ -f $OUT/ddm_ladder/ddm-d10-smoke/exp_ladder_d768/last.ckpt ] || { log "FAIL: A left no last.ckpt"; fail=1; }
train A2 900 0.1397 0 160 320 $A
grep -q "resuming from .*last" $OUT/A2.log || { log "FAIL: A2's log shows no resume"; fail=1; }
[ -f $OUT/ddm_ladder/ddm-d10-smoke/exp_ladder_d768/ms-step=000320.ckpt ] || { log "FAIL: no ddm_ladder/ddm-d10-smoke/exp_ladder_d768/ms-step=000320.ckpt"; fail=1; }
# E1: the evaluation, 2 batches per rank at 4 calls
log "START E1 (ddm at 4 calls, 2 batches per rank)"
bash $WT/jobs/evals_ddm_ladder.sh --root $OUT --limit 2 --calls "4" ddm-d10-smoke:768:10:2:320 > $OUT/E1.log 2>&1
rc=$?
grep "sample check\|^task \|^ddm ladder evaluations" $OUT/E1.log | cut -c1-400 | tee -a $OUT/phases.log
log "END E1 rc=$rc"
[ $rc -eq 0 ] || { fail=1; tail -n 30 $OUT/E1.log; tail -n 30 $OUT/raw_view/ddm_ladder/*/eval_*.log 2>/dev/null; }
[ -L $OUT/raw_view/ddm_ladder/ddm-d10-smoke/exp_ladder_d768/ms-step=000320.ckpt ] || { log "FAIL: the raw view of ddm-d10-smoke is not a link: its EMA slot does not hold the raw weights"; fail=1; }
# H: the held-out scores of the final checkpoint and the point run's energy score
log "START H (ddm held-out of 1 checkpoint and the d10 point run's energy score)"
bash $WT/jobs/heldout_ddm_ladder.sh --root $OUT --out $OUT/heldout.jsonl > $OUT/H.log 2>&1
rc=$?
log "END H rc=$rc"
[ $rc -eq 0 ] || { fail=1; tail -n 30 $OUT/H.log; }
python3 -I - $OUT/heldout.jsonl <<'PY' | tee -a $OUT/phases.log || fail=1
import json, math, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1])]
except OSError as e:
    print("ddm smoke H: FAIL", e); sys.exit(1)
fin = lambda r, k: isinstance(r.get(k), float) and math.isfinite(r[k])
kinds = sorted((r["ckpt"].split("/")[-3], bool(r.get("ddm"))) for r in rows)
ok = kinds == [("ddm-d10-smoke", True), ("xl-d10-0p25c0", False)] and all(
    fin(r, "loss_val") and fin(r, "loss_train") and fin(r, "es_val") and fin(r, "es_train") and r.get("es") is True for r in rows)
for r in rows:
    print(f"  held-out {r['ckpt'].split('/')[-3]} {r.get('weights')}: ddm {r.get('ddm')}, draws {r.get('draws')}, loss_val {r.get('loss_val')}, "
          f"es_val {r.get('es_val')}, loss_train {r.get('loss_train')}, es_train {r.get('es_train')}, {r.get('elapsed_s')} s")
print("ddm smoke H:", "PASS" if ok else f"FAIL (records {kinds})")
sys.exit(0 if ok else 1)
PY
# E2: the full evaluation at 4 calls
log "START E2 (ddm at 4 calls, 50K captions)"
bash $WT/jobs/evals_ddm_ladder.sh --root $OUT --calls "4" ddm-d10-smoke:768:10:2:320 > $OUT/E2.log 2>&1
rc=$?
grep "^task \|^ddm ladder evaluations" $OUT/E2.log | cut -c1-300 | tee -a $OUT/phases.log
log "END E2 rc=$rc"
[ $rc -eq 0 ] || { fail=1; tail -n 30 $OUT/E2.log; tail -n 20 $OUT/raw_view/ddm_ladder/ddm-d10-smoke/eval_*.log 2>/dev/null; }
python3 -I - $OUT/raw_view/ddm_ladder/ddm-d10-smoke/exp_ladder_d768/eval_000320_ddm_calls4_official <<'PY' | tee -a $OUT/phases.log || fail=1
import json, math, os, sys
d = sys.argv[1]
try:
    m = json.load(open(os.path.join(d, "metrics.json")))
    m.update(json.load(open(os.path.join(d, "metrics_inception.json"))))
    ok = m.get("n") == 50000 and all(math.isfinite(m[k]) for k in ("fd", "inception_fd", "precision", "recall", "mmd"))
    print(f"  E2 ddm 4 calls: n {m.get('n')}, DINOv2 FD {m['fd']:.1f}, Inception FD {m['inception_fd']:.2f}, precision {m['precision']:.4f}, recall {m['recall']:.4f}; "
          f"times {json.load(open(os.path.join(d, 'phase_times.json')))}")
except (OSError, KeyError, ValueError) as e:
    ok = False; print(f"  E2: no metrics ({e})")
print("ddm smoke E2:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
log "DONE $OUT fail=$fail"
exit $fail
