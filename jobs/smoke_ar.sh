#!/bin/bash
#SBATCH -J gpic-ar-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:10:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of the autoregressive model through the production path on 8 H100s: a short training of d640, a resume from its last checkpoint, and the full evaluation (the FD of a checkpoint this young means nothing; the time of the evaluation does). No W&B.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ar_run
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/ar_smoke_$TS
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles LADDER_RUN_ROOT=$OUT
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
log "START train to 800"
BENCH_OUT=$OUT/bench/first timeout -k 60 700 bash $WT/jobs/train_ladder.sh d640 640 9 2 3200 --config configs_h100/ar_train.yaml --trainer.max_steps 800 --trainer.log_every_n_steps 50 > $OUT/first.log 2>&1
rc=$?; log "END train rc=$rc"; [ $rc -eq 0 ] || exit $rc
fail=0
python3 -I - $OUT/first.log <<'PY' | tee -a $OUT/phases.log
import re, sys
s = open(sys.argv[1], errors="replace").read()
after = s.split("[warmup] done")[-1] if "[warmup] done" in s else ""
print("warm-up done:", "[warmup] done" in s, "recompiles after warm-up", after.count("__recompiles"))
sys.exit(0 if "[warmup] done" in s else 1)
PY
[ $? -eq 0 ] || fail=1
log "START resume to 1000"
BENCH_OUT=$OUT/bench/resume timeout -k 60 420 bash $WT/jobs/train_ladder.sh d640 640 9 2 3200 --config configs_h100/ar_train.yaml --trainer.max_steps 1000 --trainer.log_every_n_steps 50 > $OUT/resume.log 2>&1
rc=$?; log "END resume rc=$rc"; [ $rc -eq 0 ] || fail=1
log "START eval"
timeout -k 60 2400 bash $WT/jobs/eval_ar.sh d640 640 9 2 800 > $OUT/eval.log 2>&1
rc=$?; log "END eval rc=$rc"; [ $rc -eq 0 ] || fail=1
ls $OUT/exp_ladder_d640/eval_000800_ar/metrics.json || fail=1
cat $OUT/exp_ladder_d640/eval_000800_ar/phase_times.json 2>/dev/null | tee -a $OUT/phases.log
log "DONE $OUT fail=$fail"
exit $fail
