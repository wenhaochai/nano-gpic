#!/bin/bash
#SBATCH -J gpic-energy-smoke
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Smoke test of the energy head through the production path on 8 H100s: a short training of d640, a resume from its last checkpoint, and the full evaluation at 8 calls (the FD of a checkpoint this young means nothing). No W&B. Code: the frozen worktree gpic-wt/energy_real.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/energy_real
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q "head_width: 352" $WT/baselines/PixelGen/configs_h100/energy_train.yaml || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
OUT=$WC/runs/gpic-speed/energy_smoke_$(date +%m%d_%H%M)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles LADDER_RUN_ROOT=$OUT GPIC_IMPLICIT_PRINT=50
export GPIC_REGION_CROP=0 GPIC_SHARD_SEED=1730000000   # as jobs/run_energy.sh, so the step time reads for the real run
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
log "START train to 800"
BENCH_OUT=$OUT/bench/first timeout -k 60 600 bash $WT/jobs/train_ladder.sh d640 640 9 2 3200 --config configs_h100/energy_train.yaml --trainer.max_steps 800 --trainer.log_every_n_steps 50 > $OUT/first.log 2>&1
rc=$?; log "END train rc=$rc"; [ $rc -eq 0 ] || exit $rc
fail=0
python3 -I - $OUT/first.log <<'PY' | tee -a $OUT/phases.log
import math, re, sys
s = open(sys.argv[1], errors="replace").read()
rows = re.findall(r"\[implicit\] call (\d+): score (\S+) to_data (\S+) between (\S+)", s)
last = rows[-1] if rows else None
ok = bool(last) and all(math.isfinite(float(v)) for v in last[1:]) and "[warmup] done" in s
after = s.split("[warmup] done")[-1] if "[warmup] done" in s else ""
print("energy check:", "PASS" if ok else "FAIL", "last", last, "lines", len(rows), "recompiles after warm-up", after.count("__recompiles"))
sys.exit(0 if ok else 1)
PY
[ $? -eq 0 ] || fail=1
log "START resume to 1000"
BENCH_OUT=$OUT/bench/resume timeout -k 60 420 bash $WT/jobs/train_ladder.sh d640 640 9 2 3200 --config configs_h100/energy_train.yaml --trainer.max_steps 1000 --trainer.log_every_n_steps 50 > $OUT/resume.log 2>&1
rc=$?; log "END resume rc=$rc"; [ $rc -eq 0 ] || fail=1
log "START eval"
timeout -k 60 900 bash $WT/jobs/eval_energy.sh 640 9 2 800 8 > $OUT/eval.log 2>&1
rc=$?; log "END eval rc=$rc"; [ $rc -eq 0 ] || fail=1
ls $OUT/exp_ladder_d640/eval_000800_calls8_prompt/metrics.json || fail=1
cat $OUT/exp_ladder_d640/eval_000800_calls8_prompt/phase_times.json 2>/dev/null | tee -a $OUT/phases.log
log "DONE $OUT fail=$fail"
exit $fail
