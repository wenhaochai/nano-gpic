#!/bin/bash
#SBATCH -J gpic-gmflow2-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of the per-pixel spread (docs/gmflow.md sections 7 and 10) through the production path on 8 H100s, after jobs/smoke_gmflow.sh:
# 1. gm_k 8 with gm_spread pixel at d640: train to 400 steps (an ms checkpoint at 400), resume from last.ckpt to 500;
# 2. predictions of that checkpoint on 512 prompts with gm_ode at 8 and 4 calls (EVAL_SPREAD=pixel);
# 3. the guard: the per-pixel checkpoint must be refused as image, with the guard's own message;
# 4. a regression check of the merged sampler on a trained model: gm_ode at 8 calls on w1173's checkpoint (gm_spread image, K = 8, C0/8) through a link in this
#    smoke's directory (about 80 s; the smoke of 2026-11-05 took 75 s for it).
# No second-order sampler (owner 2026-11-06: GM-SDE 2 / GM-ODE 2 are not run). It reports s/step, the recompiles after the warm-up, the loss, s_u and the per-pixel
# spread (mean and std over pixels of log s_u), and the sample checks; it fails on a non-finite value, a spread that never leaves 0, a bad sample check or a nonzero
# exit. No W&B. Code: the frozen worktree gpic-wt/gmflow_run3 (never the development worktree, which moves), with no local changes.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/gmflow_run3
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q "def check_bound" $WT/baselines/PixelGen/src/diffusion/flow_matching/gm_spread.py 2>/dev/null || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
W1173=$WC/runs/gpic-speed/gmflow_k8_d640/exp_ladder_d640/ms-step=062336.ckpt
[ -f $W1173 ] || { echo "no $W1173"; exit 2; }
OUT=$WC/runs/gpic-speed/gmflow2_smoke_$(date +%m%d_%H%M%S)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT GMFLOW_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles LADDER_RUN_ROOT=$OUT GPIC_GM_PRINT=50
export GPIC_REGION_CROP=0 GPIC_SHARD_SEED=1730000000   # as w1173, so the step time reads for the real C0/8 run
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
check() {   # <log> <bench dir>: finite [gmflow] lines with the per-pixel spread, a spread that moved off 0, the recompiles after the warm-up, rank 0's median step time
python3 -I - "$1" "$2" <<'PY' | tee -a $OUT/phases.log
import json, math, re, statistics, sys
s = open(sys.argv[1], errors="replace").read()
F = r"([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"   # a number, not \S+: the 8 ranks print at the same time and lines run together (w1168)
rows = re.findall(r"\[gmflow\] call (\d+): loss " + F + " s_u " + F + " n_eff " + F + " max_w " + F + " logsu_pix_mean " + F + " logsu_pix_std " + F, s)
ok = bool(rows) and all(math.isfinite(float(v)) for r in rows for v in r[1:]) and "[warmup] done" in s
moved = bool(rows) and max(float(r[6]) for r in rows) > 1e-4   # zero-initialized offset head: its std over pixels must leave 0 once trained
after = s.split("[warmup] done")[-1] if "[warmup] done" in s else ""
try:
    st = [r["step_ms"] for r in json.load(open(f"{sys.argv[2]}/steps_rank0.json"))["rows"] if "step_ms" in r][20:]
    step = f"{statistics.median(st) / 1000:.4f} s/step (median of {len(st)} steps after the first 20; p90 {sorted(st)[int(0.9 * len(st))] / 1000:.4f})"
except Exception as e:
    step = f"no step times ({e!r})"
g = [float(v) for v in re.findall(r"\[gmflow\] call \d+: grad norm before the clip " + F, s)]
ok = ok and moved and bool(g) and all(math.isfinite(v) for v in g)
print("pre-clip grad norm: first", g[:3], "last", g[-3:], "share above the clip 1.0:", round(sum(v > 1 for v in g) / max(len(g), 1), 3))
print("per-pixel log s_u (call, mean over pixels, std over pixels):", [(int(r[0]), float(r[5]), float(r[6])) for r in rows[:: max(1, len(rows) // 8)]])
print("gmflow2 check:", "PASS" if ok else "FAIL", "| spread moved off 0:", moved, "| first", rows[0] if rows else None, "| last", rows[-1] if rows else None,
      "| lines", len(rows), "| recompiles after warm-up", after.count("__recompiles"), "|", step)
sys.exit(0 if ok else 1)
PY
}
fail=0
TRAIN=(--config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k 8 --model.denoiser.init_args.gm_spread pixel --trainer.log_every_n_steps 50)
log "START train to 400 (gm_spread pixel, K = 8)"
# max_steps 1600 in train_ladder.sh: ms checkpoints every 400 steps, last.ckpt every 50; --trainer.max_steps stops the launch early
BENCH_OUT=$OUT/bench/first timeout -k 60 480 bash $WT/jobs/train_ladder.sh d640 640 9 2 1600 "${TRAIN[@]}" --trainer.max_steps 400 > $OUT/first.log 2>&1
rc=$?; log "END train rc=$rc"; [ $rc -eq 0 ] || exit $rc
check $OUT/first.log $OUT/bench/first || fail=1
ls $OUT/exp_ladder_d640/ms-step=000400*.ckpt || fail=1
log "START resume to 500"
BENCH_OUT=$OUT/bench/resume timeout -k 60 300 bash $WT/jobs/train_ladder.sh d640 640 9 2 1600 "${TRAIN[@]}" --trainer.max_steps 500 > $OUT/resume.log 2>&1
rc=$?; log "END resume rc=$rc"; [ $rc -eq 0 ] || fail=1
grep -q "resuming from .*last" $OUT/resume.log || { log "resume did not start from last.ckpt"; fail=1; }
check $OUT/resume.log $OUT/bench/resume || fail=1
# predictions: 2 batches of 32 per rank = 512 prompts, 8 calls, samples checked, no FD
export EVAL_LIMIT_BATCHES=2
for C in 8 4; do
  log "START predict pixel gm_ode $C"
  EVAL_SPREAD=pixel timeout -k 30 240 bash $WT/jobs/eval_gmflow.sh 8 400 gm_ode $C > $OUT/eval_gm_ode_$C.log 2>&1
  rc=$?; log "END predict pixel gm_ode $C rc=$rc $(cat $OUT/exp_ladder_d640/eval_000400_gm_ode_calls$C/sample_check.txt 2>/dev/null) $(cat $OUT/exp_ladder_d640/eval_000400_gm_ode_calls$C/phase_times.json 2>/dev/null)"; [ $rc -eq 0 ] || fail=1
done
# the guard: the per-pixel checkpoint must be refused as image, by the guard's assert (not by a timeout or a missing file; review_gmflow2 L3)
EVAL_SPREAD=image timeout -k 30 180 bash $WT/jobs/eval_gmflow.sh 8 400 gm_ode 2 > $OUT/eval_guard.log 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -q "per-pixel offset head presence does not match EVAL_SPREAD=image" $OUT/eval_guard.log; then log "guard refused the per-pixel checkpoint as image (rc=$rc)"
else log "the guard did not refuse the per-pixel checkpoint as image with its message (rc=$rc)"; fail=1; fi
# w1173's checkpoint (K = 8, gm_spread image, C0/8) through a link, so nothing is written in its run directory
mkdir -p $OUT/w1173/exp_ladder_d640
ln -s $W1173 $OUT/w1173/exp_ladder_d640/
log "START predict w1173 gm_ode 8"
LADDER_RUN_ROOT=$OUT/w1173 timeout -k 30 240 bash $WT/jobs/eval_gmflow.sh 8 62336 gm_ode 8 > $OUT/eval_w1173_gm_ode.log 2>&1
rc=$?; log "END predict w1173 gm_ode 8 rc=$rc $(cat $OUT/w1173/exp_ladder_d640/eval_062336_gm_ode_calls8/sample_check.txt 2>/dev/null) $(cat $OUT/w1173/exp_ladder_d640/eval_062336_gm_ode_calls8/phase_times.json 2>/dev/null)"; [ $rc -eq 0 ] || fail=1
grep -h "__recompiles" $OUT/exp_ladder_d640/eval_*/predict.log $OUT/w1173/exp_ladder_d640/eval_*/predict.log 2>/dev/null | grep -v qwen3 | head -5 | tee -a $OUT/phases.log
log "DONE $OUT fail=$fail"
exit $fail
