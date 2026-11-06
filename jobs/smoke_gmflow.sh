#!/bin/bash
#SBATCH -J gpic-gmflow-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of the GMFlow head (docs/gmflow.md) through the production path on 8 H100s, at gm_k = 8: compile, DDP, a short training of d640 with an ms checkpoint, a resume from
# last.ckpt, then predictions on 512 prompts with GM-SDE, GM-ODE and Euler on the mean (8 calls), and GM-SDE as DDPM-small on w1147's point model. It reports the recompiles after
# the warm-up, s/step, and fails on a non-finite loss, s_u or effective number of components. No W&B. Code: this worktree, gpic-wt/gmflow, with no local changes.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/gmflow
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q "gm_k: 8" $WT/baselines/PixelGen/configs_h100/gmflow_train.yaml || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/gmflow_smoke_$(date +%m%d_%H%M%S)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles LADDER_RUN_ROOT=$OUT GPIC_GM_PRINT=50
export GPIC_REGION_CROP=0 GPIC_SHARD_SEED=1730000000   # as jobs/run_gmflow.sh, so the step time reads for the real run
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
check() {   # <log> <bench dir>: finite [gmflow] lines after the warm-up, the recompiles after it, the median step time of rank 0
python3 -I - "$1" "$2" <<'PY' | tee -a $OUT/phases.log
import json, math, re, statistics, sys
s = open(sys.argv[1], errors="replace").read()
F = r"([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"   # a number, not \S+: the 8 ranks print at the same time and lines run together (w1168: "max_w 0.127[gmflow] call 50 ...")
rows = re.findall(r"\[gmflow\] call (\d+): loss " + F + " s_u " + F + " n_eff " + F + " max_w " + F, s)
ok = bool(rows) and all(math.isfinite(float(v)) for r in rows for v in r[1:]) and "[warmup] done" in s
after = s.split("[warmup] done")[-1] if "[warmup] done" in s else ""
try:
    st = [r["step_ms"] for r in json.load(open(f"{sys.argv[2]}/steps_rank0.json"))["rows"] if "step_ms" in r][20:]
    step = f"{statistics.median(st) / 1000:.4f} s/step (median of {len(st)} steps after the first 20)"
except Exception as e:
    step = f"no step times ({e!r})"
g = [float(v) for v in re.findall(r"\[gmflow\] call \d+: grad norm before the clip " + F, s)]
ok = ok and bool(g) and all(math.isfinite(v) for v in g)
print("pre-clip grad norm: first", g[:3], "last", g[-3:], "share above the clip 1.0:", sum(v > 1 for v in g) / max(len(g), 1))
print("gmflow check:", "PASS" if ok else "FAIL", "| first", rows[0] if rows else None, "| last", rows[-1] if rows else None, "| lines", len(rows),
      "| recompiles after warm-up", after.count("__recompiles"), "|", step)
sys.exit(0 if ok else 1)
PY
}
fail=0
log "START train to 800"
BENCH_OUT=$OUT/bench/first timeout -k 60 600 bash $WT/jobs/train_ladder.sh d640 640 9 2 3200 --config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k 8 --trainer.max_steps 800 --trainer.log_every_n_steps 50 > $OUT/first.log 2>&1
rc=$?; log "END train rc=$rc"; [ $rc -eq 0 ] || exit $rc
check $OUT/first.log $OUT/bench/first || fail=1
ls $OUT/exp_ladder_d640/ms-step=000800*.ckpt || fail=1
log "START resume to 1000"
BENCH_OUT=$OUT/bench/resume timeout -k 60 420 bash $WT/jobs/train_ladder.sh d640 640 9 2 3200 --config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k 8 --trainer.max_steps 1000 --trainer.log_every_n_steps 50 > $OUT/resume.log 2>&1
rc=$?; log "END resume rc=$rc"; [ $rc -eq 0 ] || fail=1
grep -q "resuming from .*last" $OUT/resume.log || { log "resume did not start from last.ckpt"; fail=1; }
check $OUT/resume.log $OUT/bench/resume || fail=1
# predictions: 2 batches of 32 per rank = 512 prompts, 8 calls, samples checked, no FD
export EVAL_LIMIT_BATCHES=2
for S in gm_sde gm_ode euler_mean; do
  log "START predict $S"
  timeout -k 60 360 bash $WT/jobs/eval_gmflow.sh 8 800 $S 8 > $OUT/eval_$S.log 2>&1
  rc=$?; log "END predict $S rc=$rc $(cat $OUT/exp_ladder_d640/eval_000800_${S}_calls8/sample_check.txt 2>/dev/null) $(cat $OUT/exp_ladder_d640/eval_000800_${S}_calls8/phase_times.json 2>/dev/null)"; [ $rc -eq 0 ] || fail=1
done
# the point model with GM-SDE (DDPM-small): w1147's checkpoint through a link, so nothing is written in its run directory
mkdir -p $OUT/point/exp_ladder_d640
ln -s $WC/runs/gpic-speed/point_nodrop_d640/exp_ladder_d640/ms-step=033408.ckpt $OUT/point/exp_ladder_d640/
log "START predict point gm_sde"
LADDER_RUN_ROOT=$OUT/point timeout -k 60 360 bash $WT/jobs/eval_gmflow.sh 0 33408 gm_sde 8 > $OUT/eval_point.log 2>&1
rc=$?; log "END predict point rc=$rc $(cat $OUT/point/exp_ladder_d640/eval_033408_gm_sde_calls8/sample_check.txt 2>/dev/null)"; [ $rc -eq 0 ] || fail=1
grep -h "__recompiles" $OUT/exp_ladder_d640/eval_*/predict.log 2>/dev/null | head -5 | tee -a $OUT/phases.log
log "DONE $OUT fail=$fail"
exit $fail
