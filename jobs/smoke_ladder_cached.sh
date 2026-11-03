#!/bin/bash
#SBATCH -J gpic-ladder-smoke3
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The text cache on 8 H100s, with the real job scripts and the frozen worktree gpic-wt/ladder_cache:
# 1. stage timing at d640 (600 steps, ms checkpoint at step 600), d1024, d1280 (400 steps each), against w953's online path;
# 2. resume: d640 from a last.ckpt of the online path (step 320 of w920) to step 400;
# 3. epoch ends: d640 on 16 shards with 2 workers per rank for 2400 steps: shard switches and 3 passes on the cached path;
# 4. the eval path (online encoder) on the checkpoint of arm 1;
# 5. d640 with the kernels launched from C++ and with CUDA graphs in the denoiser: without the encoder, a small model's step
#    may be limited by the host launching kernels. Last, so a stop at the time limit costs nothing else.
# Exit 1 when a part fails. No W&B.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_cache
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/ladder_smoke3_$TS
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles
fail=0
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }
arm() {   # arm <tag> <size> <d> <nb> <nt> <max_steps of the run> <steps of this launch> [main.py arguments...]
  local tag=$1 size=$2 d=$3 nb=$4 nt=$5 steps=$6 stop=$7; shift 7
  log "START $tag"
  BENCH_OUT=$OUT/bench/$tag BENCH_STAGES=1 LADDER_RUN_ROOT=$OUT/$tag timeout 900 \
    bash $WT/jobs/train_ladder.sh $size $d $nb $nt $steps --trainer.max_steps $stop "$@" > $OUT/$tag.log 2>&1
  local rc=$?
  log "END $tag rc=$rc"; [ $rc -eq 0 ] || fail=1
}
arm d640 d640 640 9 2 2400 600
arm d1024 d1024 1024 12 3 32000 400
arm d1280 d1280 1280 14 3 32000 400
mkdir -p $OUT/resume/exp_ladder_d640
cp $WC/runs/gpic-speed/ladder_smoke_1102_1636/exp_ladder_d640/last.ckpt $OUT/resume/exp_ladder_d640/last.ckpt
arm resume d640 640 9 2 3200 400
grep -q "resuming from" $OUT/resume.log || { log "resume: no resume line"; fail=1; }
arm epoch d640 640 9 2 32000 2400 --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..00015}.tar' --data.train_num_workers 2
log "START eval"
LADDER_RUN_ROOT=$OUT/d640 bash $WT/jobs/eval_ladder.sh d640 640 9 2 600 > $OUT/eval.log 2>&1; rc=$?
log "END eval rc=$rc"; [ $rc -eq 0 ] || fail=1
required=$fail   # these two only measure: a failure here does not fail the smoke
TORCHINDUCTOR_CPP_WRAPPER=1 arm d640_cpp d640 640 9 2 32000 600
GPIC_PROF_DEN_COMPILE_MODE=reduce-overhead arm d640_graphs d640 640 9 2 32000 1200
fail=$required
python $WT/tools/prof/ladder_stages.py $OUT 2>&1 | tee -a $OUT/phases.log
python - $OUT <<'PY' 2>&1 | tee -a $OUT/phases.log
import json, re, sys
out = sys.argv[1]
first = json.load(open(f"{out}/bench/resume/steps_rank0.json"))["rows"][0]["step"]
print("resume: first timed step", first, "(320 expected)")
elog = open(f"{out}/epoch.log", errors="replace").read()
passes = {}
for rank, p, shards in re.findall(r"\[shards\] rank (\d+) pass (\d+): (.+)", elog):
    passes.setdefault(int(p), {})[int(rank)] = shards.strip()
full = {p: v for p, v in passes.items() if len(v) == 8}
print("epoch run:", len(full), "passes by all 8 ranks; same shards", all(len(set(v.values())) == 1 for v in full.values()))
print("dropped samples or unknown keys in the epoch run:", len(re.findall(r"KeyError|exception|Traceback", elog)))
PY
log "DONE fail=$fail $OUT"
exit $fail
