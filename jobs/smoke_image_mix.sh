#!/bin/bash
#SBATCH -J gpic-image-mix-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of one mixture per image (gm_scope image, docs/gmflow.md section 15) through the production path on 8 H100s:
# 1. gm_k 8 with gm_scope image at d640 on the data of w1173 (shard seed 1730000000, the official crop): train to 400 steps (an ms checkpoint at 400), resume from
#    last.ckpt to 500. The saved config must hold gm_scope image; loss, s_u, n_eff, max_w and the pre-clip grad norms must be finite; win_max, win_alive and w_dead
#    (the largest share of a rank's batch that one candidate wins, the candidates that win at least one image, the weight on the candidates that win none) are
#    printed every 50 calls;
# 2. predictions of that checkpoint on 512 prompts: gm_draw at 8 and 50 calls and euler_mean at 8 calls (EVAL_SCOPE=image; this root's scope.txt is image);
# 3. the guards: EVAL_SCOPE=pixel on this root, and gm_ode with EVAL_SCOPE=image, are refused with their messages;
# 4. a regression check of the per-pixel path on this code: gm_draw at 8 calls on w1173's checkpoint (K = 8, per pixel, C0/8) through a link, so nothing is
#    written in its run directory.
# It reports s/step and the recompiles after the warm-up; it fails on a non-finite value, a missing sample check, a guard that does not refuse, or a nonzero exit.
# No W&B. Code: the frozen worktree named by the first argument (default gpic-wt/gmflow_run6), with no local changes.
# usage: smoke_image_mix.sh [worktree name under gpic-wt/]
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/${1:-gmflow_run6}
[ -e "$WT/.git" ] || { echo "no worktree $WT"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
grep -q "def gm_pick" $WT/baselines/PixelGen/src/diffusion/flow_matching/gmflow.py || { echo "$WT holds old code"; exit 2; }
W1173=$WC/runs/gpic-speed/gmflow_k8_d640/exp_ladder_d640/ms-step=062336.ckpt
[ -f $W1173 ] || { echo "no $W1173"; exit 2; }
OUT=$WC/runs/gpic-speed/image_mix_smoke_$(date +%m%d_%H%M%S)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
echo image > $OUT/scope.txt   # eval_gmflow.sh checks the root's scope; train_gmflow.sh writes it for a real run
export LADDER_WT=$WT GMFLOW_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles LADDER_RUN_ROOT=$OUT GPIC_GM_PRINT=50
export GPIC_REGION_CROP=0 GPIC_SHARD_SEED=1730000000   # as w1173, so the step time reads for the real C0/8 run
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
check() {   # <log> <bench dir>: finite [gmflow] lines with the win counts, finite grad norms, the recompiles after the warm-up, rank 0's median step time
python3 -I - "$1" "$2" <<'PY' | tee -a $OUT/phases.log
import json, math, re, statistics, sys
s = open(sys.argv[1], errors="replace").read()
F = r"([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"   # a number, not \S+: the 8 ranks print at the same time and lines run together (w1168)
rows = re.findall(r"\[gmflow\] call (\d+): loss " + F + " s_u " + F + " n_eff " + F + " max_w " + F + " win_max " + F + r" win_alive (\d+) w_dead " + F, s)
g = [float(v) for v in re.findall(r"\[gmflow\] call \d+: grad norm before the clip " + F, s)]
ok = bool(rows) and all(math.isfinite(float(v)) for r in rows for v in r[1:]) and bool(g) and all(math.isfinite(v) for v in g) and "[warmup] done" in s
after = s.split("[warmup] done")[-1] if "[warmup] done" in s else ""
try:
    st = [r["step_ms"] for r in json.load(open(f"{sys.argv[2]}/steps_rank0.json"))["rows"] if "step_ms" in r][20:]
    step = f"{statistics.median(st) / 1000:.4f} s/step (median of {len(st)} steps after the first 20; p90 {sorted(st)[int(0.9 * len(st))] / 1000:.4f})"
except Exception as e:
    step = f"no step times ({e!r})"
print("pre-clip grad norm: first", g[:3], "last", g[-3:], "share above the clip 1.0:", round(sum(v > 1 for v in g) / max(len(g), 1), 3))
print("(call, loss, s_u, n_eff, max_w, win_max, win_alive, w_dead):", [(int(r[0]), *map(float, r[1:6]), int(r[6]), float(r[7])) for r in rows[:: max(1, len(rows) // 8)]])
print("image-mix check:", "PASS" if ok else "FAIL", "| lines", len(rows), "| recompiles after warm-up", after.count("__recompiles"), "|", step)
sys.exit(0 if ok else 1)
PY
}
fail=0
TRAIN=(--config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k 8 --model.denoiser.init_args.gm_scope image --trainer.log_every_n_steps 50)
log "START train to 400 (K = 8, gm_scope image)"
# max_steps 1600 in train_ladder.sh: ms checkpoints every 400 steps, last.ckpt every 50; --trainer.max_steps stops the launch early
BENCH_OUT=$OUT/bench/first timeout -k 60 480 bash $WT/jobs/train_ladder.sh d640 640 9 2 1600 "${TRAIN[@]}" --trainer.max_steps 400 > $OUT/first.log 2>&1
rc=$?; log "END train rc=$rc"; [ $rc -eq 0 ] || exit $rc
check $OUT/first.log $OUT/bench/first || fail=1
ls $OUT/exp_ladder_d640/ms-step=000400*.ckpt || fail=1
grep -q "gm_scope: image" $OUT/exp_ladder_d640/config-*.yaml 2>/dev/null || { log "gm_scope: image is not in the saved config"; fail=1; }
log "START resume to 500"
BENCH_OUT=$OUT/bench/resume timeout -k 60 300 bash $WT/jobs/train_ladder.sh d640 640 9 2 1600 "${TRAIN[@]}" --trainer.max_steps 500 > $OUT/resume.log 2>&1
rc=$?; log "END resume rc=$rc"; [ $rc -eq 0 ] || fail=1
grep -q "resuming from .*last" $OUT/resume.log || { log "resume did not start from last.ckpt"; fail=1; }
check $OUT/resume.log $OUT/bench/resume || fail=1
# predictions: 2 batches of 32 per rank = 512 prompts, samples checked, no FD
export EVAL_LIMIT_BATCHES=2
for spec in "gm_draw 8" "gm_draw 50" "euler_mean 8"; do
  read -r S C <<< "$spec"
  log "START predict image $S $C"
  EVAL_SCOPE=image timeout -k 30 300 bash $WT/jobs/eval_gmflow.sh 8 400 $S $C > $OUT/eval_${S}_$C.log 2>&1
  rc=$?
  D=$OUT/exp_ladder_d640/eval_000400_${S}_calls${C}_official
  chk=$(cat $D/sample_check.txt 2>/dev/null)
  log "END predict image $S $C rc=$rc sample_check: $chk $(cat $D/phase_times.json 2>/dev/null)"
  [ $rc -eq 0 ] && [ -n "$chk" ] || fail=1
done
# the guards refuse with their own messages, before any model is loaded
EVAL_SCOPE=pixel timeout -k 30 120 bash $WT/jobs/eval_gmflow.sh 8 400 gm_draw 2 > $OUT/eval_guard_scope.log 2>&1
rc=$?
if [ $rc -eq 2 ] && grep -q "the run was trained with scope image, EVAL_SCOPE is pixel" $OUT/eval_guard_scope.log; then log "guard: scope pixel refused on an image root (rc=$rc)"
else log "guard: scope pixel was not refused with its message (rc=$rc)"; fail=1; fi
EVAL_SCOPE=image timeout -k 30 120 bash $WT/jobs/eval_gmflow.sh 8 400 gm_ode 2 > $OUT/eval_guard_ode.log 2>&1
rc=$?
if [ $rc -eq 2 ] && grep -q "is evaluated with gm_draw or euler_mean" $OUT/eval_guard_ode.log; then log "guard: gm_ode refused for one mixture per image (rc=$rc)"
else log "guard: gm_ode was not refused with its message (rc=$rc)"; fail=1; fi
# the per-pixel path on this code: w1173's checkpoint through a link (its root has no scope.txt, so it is per pixel)
mkdir -p $OUT/w1173/exp_ladder_d640
ln -s $W1173 $OUT/w1173/exp_ladder_d640/
log "START predict w1173 gm_draw 8"
LADDER_RUN_ROOT=$OUT/w1173 timeout -k 30 300 bash $WT/jobs/eval_gmflow.sh 8 62336 gm_draw 8 > $OUT/eval_w1173_gm_draw.log 2>&1
rc=$?
D=$OUT/w1173/exp_ladder_d640/eval_062336_gm_draw_calls8_official
chk=$(cat $D/sample_check.txt 2>/dev/null)
log "END predict w1173 gm_draw 8 rc=$rc sample_check: $chk $(cat $D/phase_times.json 2>/dev/null)"
[ $rc -eq 0 ] && [ -n "$chk" ] || fail=1
grep -h "__recompiles" $OUT/exp_ladder_d640/eval_*/predict.log $OUT/w1173/exp_ladder_d640/eval_*/predict.log 2>/dev/null | grep -v qwen3 | head -5 | tee -a $OUT/phases.log
log "DONE $OUT fail=$fail"
exit $fail
