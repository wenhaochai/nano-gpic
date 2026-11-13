#!/bin/bash
#SBATCH -J gpic-implicit-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of the implicit sampler, end to end, on 8 H100s. A short training of d640 with the overlay, a resume from its last checkpoint, and
# the evaluation path at 1 and 8 calls (the FD of a checkpoint this young means nothing). No W&B.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/dist_step_run
[ -d "$WT" ] || { echo "no worktree $WT"; exit 2; }
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/implicit_smoke_$TS
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles LADDER_RUN_ROOT=$OUT IMPLICIT_NAME=smoke IMPLICIT_STEPS=3200 GPIC_IMPLICIT_PRINT=100
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }

log "START train to 800"
BENCH_OUT=$OUT/bench/first timeout -k 60 600 bash $WT/jobs/train_implicit.sh --trainer.max_steps 800 --trainer.log_every_n_steps 50 > $OUT/first.log 2>&1
log "END train rc=$?"
# the checks of review finding 6: a finite score, samples that separate, no recompile after the warm-up
python3 -I - $OUT/first.log <<'PY' | tee -a $OUT/phases.log
import math, re, sys
s = open(sys.argv[1], errors="replace").read()
rows = re.findall(r"\[implicit\] call (\d+): score (\S+) to_data (\S+) between (\S+)", s)
last = rows[-1] if rows else None
ok = bool(last) and all(math.isfinite(float(v)) for v in last[1:]) and float(last[3]) > 1e-3
after = s.split("[warmup] done")[-1] if "[warmup] done" in s else ""
print("implicit check:", "PASS" if ok else "FAIL", "last", last, "lines", len(rows), "recompiles after warm-up", after.count("__recompiles"))
PY
log "START resume to 1000"
BENCH_OUT=$OUT/bench/resume timeout -k 60 420 bash $WT/jobs/train_implicit.sh --trainer.max_steps 1000 --trainer.log_every_n_steps 50 > $OUT/resume.log 2>&1
log "END resume rc=$?"
log "START eval at 8 and 1 calls"
timeout -k 60 900 bash $WT/jobs/eval_implicit.sh smoke 640 9 2 800 8 1 > $OUT/eval.log 2>&1
log "END eval rc=$?"
log "DONE $OUT"
