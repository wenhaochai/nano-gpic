#!/bin/bash
#SBATCH -J gpic-artm-ladder-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of ARTM on the final ladder's trainer (docs/artm_ladder.md; owner popups 2026-11-13: ARTM-3 at d10, d12, d14, approved after this test), on 8 H100s,
# with the production scripts of the frozen worktree gpic-wt/artm_ladder_run in a root of this smoke:
#   A   jobs/train_artm_ladder.sh at d10 (768 x (10+2), DDP, SDPA attention, micro-batch 64 x 2): a run of 320 steps, stopped at step 160;
#   A2  the same command without the stop: it resumes from last.ckpt at step 160 and ends at step 320 with ms-step=000320.ckpt;
#   C   d14 (1280 x (14+3), FSDP2 group 7, micro-batch 64 x 2): 320 steps in one piece;
#   K   tools/artm_ladder/kv_check.py on A2's checkpoint: the KV-cached sampling against the full masked forward, in bf16, with and without the fused norm;
#   E1  jobs/evals_artm_ladder.sh --limit 2 on C's checkpoint: the sample check;
#   E2  jobs/evals_artm_ladder.sh in full on A2's checkpoint: 50K captions, both metric files, the predict time.
# It fails on a nonzero exit, a non-finite loss, step files of other than 8 ranks, a peak above 74 GiB, a GPU time per optimizer step over 1.3 times the
# estimate (the point run's step time x ARTM's FLOP ratio x 1.2 for the SDPA path: 0.0762 x 2.128 x 1.2 at d10, 0.2017 x 1.992 x 1.2 at d14), an NCCL other
# than 2.27.7, the wrong strategy, no '[warmup] done', a missing checkpoint, a raw view that is not a link, a failed KV check, sample check or metric file.
# usage: smoke_artm_ladder.sh
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/artm_ladder_run
TR=$WT/jobs/train_artm_ladder.sh
[ -f $TR ] && [ -f $WT/jobs/evals_artm_ladder.sh ] || { echo "$WT lacks the ARTM job scripts"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/artm_ladder_smoke_$(date +%m%d_%H%M%S)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
check() {   # <arm> <log> <bench dir> <estimated s/step> <FSDP2 expected: 0|1> <first step expected> <last step expected> <micro-batches per step>
python3 -I - "$@" <<'PY' | tee -a $OUT/phases.log
import glob, json, math, re, statistics, sys
arm, logf, bench, est, fsdp, first, last = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5] == "1", int(sys.argv[6]), int(sys.argv[7])
ACC = int(sys.argv[8])   # the bench rows are micro-batches
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
gmed = statistics.median(gp) / 1000 * ACC if gp else float("nan")   # per optimizer step, as the estimate is
smed = statistics.median(st) / 1000 * ACC if st else float("nan")
if not gmed <= 1.3 * est: bad.append(f"{gmed:.4f} s of GPU time per step is over 1.3 x the estimate {est}")
nccl = re.findall(r"NCCL version ([0-9][0-9.]*)", s)
if not nccl or any(v != "2.27.7" for v in nccl): bad.append(f"NCCL versions {sorted(set(nccl))}")
n_fsdp = s.count("FSDP2: blocks sharded")
if (n_fsdp > 0) != fsdp: bad.append(f"{n_fsdp} FSDP2 lines, FSDP2 expected: {fsdp}")
if "[warmup] done" not in s: bad.append("no '[warmup] done'")
steps = [x for x, _ in r0]
if not steps or min(steps) > first + 2 or max(steps) < last - 2: bad.append(f"rank 0 steps {steps[:1]}..{steps[-1:]}, expected {first}..{last}")
after = s.split("[warmup] done")[-1] if "[warmup] done" in s else ""
print(f"artm smoke {arm}:", "PASS" if not bad else "FAIL " + "; ".join(bad))
print(f"  {arm}: GPU time {gmed:.4f} s/step (estimate {est}, ratio {gmed / est:.3f}), start to start {smed:.4f} s/step; peak reserved {peak:.1f} GiB; NCCL {sorted(set(nccl))}; "
      f"FSDP2 lines {n_fsdp}; recompiles after the warm-up {after.count('__recompiles')}; start-up (s) {head}")
print(f"  {arm}: loss on rank 0 (step, loss): first {r0[:2]}, middle {r0[len(r0) // 2: len(r0) // 2 + 2]}, last {r0[-2:]}")
sys.exit(1 if bad else 0)
PY
}
fail=0
train() {   # <arm> <inner limit s> <estimated s/step> <FSDP2: 0|1> <first step> <last step> <the 11 list columns> [main.py args]
  local n=$1 lim=$2 est=$3 fs=$4 s0=$5 s1=$6; shift 6
  local prod=$WC/runs/gpic-speed/artm_ladder/$1 before=0
  [ -e $prod ] && before=1
  log "START $n: $*"
  SWEEP_RUN_ROOT=$OUT/artm_ladder/$1 BENCH_OUT=$OUT/bench/$n timeout -k 60 $lim bash $TR "$@" > $OUT/$n.log 2>&1
  local rc=$?
  log "END $n rc=$rc"
  [ $rc -eq 0 ] || { fail=1; tail -n 40 $OUT/$n.log; }
  check $n $OUT/$n.log $OUT/bench/$n $est $fs $s0 $s1 2 || fail=1
  if [ $before -eq 0 ] && [ -e $prod ]; then log "FAIL: the smoke created $prod"; fail=1; fi
}
A="artm-d10-smoke 768 10 2 320 1.360e-02 0.000501 0.03682 full 1024 50"
train A  900 0.1946 0 0 160 $A --trainer.max_steps 160
[ -f $OUT/artm_ladder/artm-d10-smoke/exp_ladder_d768/last.ckpt ] || { log "FAIL: A left no last.ckpt"; fail=1; }
train A2 900 0.1946 0 160 320 $A
grep -q "resuming from .*last" $OUT/A2.log || { log "FAIL: A2's log shows no resume"; fail=1; }
train C 1500 0.4822 1 0 320 artm-d14-smoke 1280 14 3 320 3.400e-03 0.000238 0.03271 full 1024 50
for r in "artm-d10-smoke d768" "artm-d14-smoke d1280"; do
  set -- $r
  [ -f $OUT/artm_ladder/$1/exp_ladder_$2/ms-step=000320.ckpt ] || { log "FAIL: no artm_ladder/$1/exp_ladder_$2/ms-step=000320.ckpt"; fail=1; }
done
# K: the KV cache on the GPU, with and without the fused norm
for fn in 1 0; do
  log "START K (fused norm $fn)"
  ( cd $WT/baselines/PixelGen && PYTHONPATH=$PWD GPIC_MFU_FUSED_NORM=$fn python $WT/tools/artm_ladder/kv_check.py $OUT/artm_ladder/artm-d10-smoke/exp_ladder_d768/ms-step=000320.ckpt 4 ) > $OUT/K$fn.log 2>&1
  rc=$?
  grep "^transition\|^kv check" $OUT/K$fn.log | tee -a $OUT/phases.log
  log "END K (fused norm $fn) rc=$rc"
  [ $rc -eq 0 ] || { fail=1; tail -n 20 $OUT/K$fn.log; }
done
# E1: the sample check on the FSDP2-trained checkpoint
log "START E1 (artm, 2 batches per rank: d14)"
bash $WT/jobs/evals_artm_ladder.sh --root $OUT --limit 2 artm-d14-smoke:1280:14:3:320 > $OUT/E1.log 2>&1
rc=$?
grep "sample check\|^task \|^artm ladder evaluations" $OUT/E1.log | cut -c1-400 | tee -a $OUT/phases.log
log "END E1 rc=$rc"
[ $rc -eq 0 ] || { fail=1; tail -n 30 $OUT/E1.log; tail -n 30 $OUT/raw_view/artm_ladder/*/eval_*.log 2>/dev/null; }
[ -L $OUT/raw_view/artm_ladder/artm-d14-smoke/exp_ladder_d1280/ms-step=000320.ckpt ] || { log "FAIL: the raw view of artm-d14-smoke is not a link"; fail=1; }
# E2: the full evaluation at d10 (smoke w2642, 2026-11-13: 50K captions took 23:06 to predict on 8 H100s, so 1500 s cut the FD; now 2400 s)
log "START E2 (d10: artm on 50K captions)"
timeout -k 60 2400 bash $WT/jobs/evals_artm_ladder.sh --root $OUT artm-d10-smoke:768:10:2:320 > $OUT/E2.log 2>&1
rc=$?
grep "^task \|^artm ladder evaluations" $OUT/E2.log | cut -c1-300 | tee -a $OUT/phases.log
log "END E2 rc=$rc"
[ $rc -eq 0 ] || { fail=1; tail -n 30 $OUT/E2.log; tail -n 20 $OUT/raw_view/artm_ladder/artm-d10-smoke/eval_*.log 2>/dev/null; }
python3 -I - $OUT <<'PY' | tee -a $OUT/phases.log || fail=1
import json, math, os, sys
d = f"{sys.argv[1]}/raw_view/artm_ladder/artm-d10-smoke/exp_ladder_d768/eval_000320_artm_calls3_official"
try:
    m = json.load(open(os.path.join(d, "metrics.json"))); m.update(json.load(open(os.path.join(d, "metrics_inception.json"))))
    ok = m.get("n") == 50000 and all(math.isfinite(m[k]) for k in ("fd", "inception_fd", "precision", "recall"))
    print(f"  E2 artm: n {m.get('n')}, DINOv2 FD {m['fd']:.1f}, Inception FD {m['inception_fd']:.2f}, precision {m['precision']:.4f}, recall {m['recall']:.4f}; "
          f"times {json.load(open(os.path.join(d, 'phase_times.json')))}")
except (OSError, KeyError, ValueError) as e:
    ok = False; print(f"  E2 artm: no metrics ({e})")
print("artm smoke E2:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
log "DONE $OUT fail=$fail"
exit $fail
