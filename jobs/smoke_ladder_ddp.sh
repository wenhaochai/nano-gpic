#!/bin/bash
#SBATCH -J gpic-ladder-smoke2
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The review fixes on 8 GPUs (gpic-wt/ladder_run): d1280 for 320 steps with the warm-up compiled inside DDP's forward
# context (DDPOptimizer on), to compare its step time with the warm-up outside it;
# then the eval path with the size check. Exit 1 when a part fails.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_run
TS=$(date +%m%d_%H%M)
SMK=$WC/runs/gpic-speed/ladder_smoke2_$TS
mkdir -p $SMK
git -C $WT rev-parse HEAD > $SMK/commit.txt
export LADDER_WT=$WT LADDER_RUN_ROOT=$SMK TORCH_LOGS=recompiles
fail=0
log() { echo "[$(date +%T)] $*" | tee -a $SMK/phases.log; }
log "START d1280"
BENCH_OUT=$SMK/bench/d1280 bash $WT/jobs/train_ladder.sh d1280 1280 14 3 320 --trainer.logger.init_args.name ladder_smoke2 \
  --trainer.logger.init_args.id ladder-smoke2-$TS > $SMK/d1280.log 2>&1; rc=$?
log "END d1280 rc=$rc"; [ $rc -eq 0 ] || fail=1
log "START eval"
bash $WT/jobs/eval_ladder.sh d1280 1280 14 3 320 > $SMK/eval.log 2>&1; rc=$?
log "END eval rc=$rc"; [ $rc -eq 0 ] || fail=1
python - $SMK/bench/d1280/steps_rank0.json <<'PY' 2>&1 | tee -a $SMK/phases.log
import json, statistics, sys
d = json.load(open(sys.argv[1]))
s = [r["step_ms"] for r in d["rows"] if "step_ms" in r]
print("d1280 median step ms, last 100:", round(statistics.median(s[-100:]), 1), "; all:", round(statistics.median(s), 1), "; graphs:", d["compile"]["counters"]["stats"]["unique_graphs"])
PY
log "DONE fail=$fail $SMK"
exit $fail
