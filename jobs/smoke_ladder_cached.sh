#!/bin/bash
#SBATCH -J gpic-ladder-smoke3
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The text cache on 8 H100s, with the real job scripts and the frozen worktree gpic-wt/ladder_cache:
# 1. stage timing at d640 (600 steps, ms checkpoint at step 600), d1024, d1280 (400 steps each), against w953's online path;
# 2. resume: d640 from a last.ckpt of the online path (step 320 of w920) to step 400;
# 3. epoch ends: d640 on 16 shards with 2 workers per rank for 2400 steps: shard switches and 3 passes on the cached path;
# 4. the eval path (online encoder) on the checkpoint of arm 1;
# 5. d640 for 3400 steps, three worker windows, with the image work after grouping and, from the worktree ladder_cache_old
#    (151970f), with the images decoded in the window: the same arm of each, so the difference is the data path;
# 6. d640 with the kernels launched from C++ and with CUDA graphs in the denoiser: without the encoder, a small model's step
#    may be limited by the host launching kernels.
# Parts 5 and 6 only measure, and they run after the summary of parts 1-4, so a stop at the time limit costs nothing else.
# Exit 1 when a part of 1-4 fails or its checks fail. No W&B.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_cache
[ -d "$WT" ] || { echo "no worktree $WT"; exit 2; }
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
  BENCH_OUT=$OUT/bench/$tag BENCH_STAGES=1 LADDER_RUN_ROOT=$OUT/$tag timeout ${ARM_LIMIT:-900} \
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
# the same arm took 12.6 min on the online path and is bound by data
ARM_LIMIT=1500 arm epoch d640 640 9 2 32000 2400 --data.train_dataset.init_args.tar_pattern 'gpic_train_{00000..00015}.tar' --data.train_num_workers 2
log "START eval"
LADDER_RUN_ROOT=$OUT/d640 timeout 1200 bash $WT/jobs/eval_ladder.sh d640 640 9 2 600 > $OUT/eval.log 2>&1; rc=$?
log "END eval rc=$rc"; [ $rc -eq 0 ] || fail=1
python $WT/tools/prof/ladder_stages.py $OUT 2>&1 | tee -a $OUT/phases.log
python - $OUT <<'PY' 2>&1 | tee -a $OUT/phases.log
import json, re, sys
out = sys.argv[1]
bad = 0


def check(name, f):
    """One check: a missing file fails this check only, and the others still print."""
    global bad
    try:
        good, text = f()
    except Exception as e:
        good, text = False, repr(e)
    bad += not good
    print(("ok   " if good else "FAIL ") + name + ": " + text)


def first_step():
    first = json.load(open(f"{out}/bench/resume/steps_rank0.json"))["rows"][0]["step"]
    return first == 320, f"first timed step {first} (320 expected)"


def passes():
    elog = open(f"{out}/epoch.log", errors="replace").read()
    seen = {}
    for rank, p, shards in re.findall(r"\[shards\] rank (\d+) pass (\d+): (.+)", elog):
        seen.setdefault(int(p), {})[int(rank)] = shards.strip()
    full = {p: v for p, v in seen.items() if len(v) == 8}
    same = all(len(set(v.values())) == 1 for v in full.values())
    return len(full) >= 3 and same, f"{len(full)} passes by all 8 ranks (3 expected), same shards on all ranks: {same}"


def drops(tag):
    def f():
        elog = open(f"{out}/{tag}.log", errors="replace").read()
        n = len(re.findall(r"webdataset/handlers\.py|KeyError|Traceback|repeating the sample before it", elog))
        return n == 0, f"{n} dropped or repeated samples or errors"
    return f


def recompiles(tag):
    def f():
        elog = open(f"{out}/{tag}.log", errors="replace").read()
        return True, f"{len(re.findall(r'Recompiling function', elog))} recompile lines"
    return f


check("resume", first_step)
check("epoch run", passes)
for tag in ("d640", "d1024", "d1280", "resume", "epoch"):
    check(f"{tag} data path", drops(tag))
    check(f"{tag} kernels", recompiles(tag))
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || fail=1
required=$fail   # the arms below only measure: a failure there does not fail the smoke
log "PARTS 1-4 fail=$fail"
arm d640_long d640 640 9 2 32000 3400
LADDER_WT=$WC/project/gpic-wt/ladder_cache_old arm d640_long_old d640 640 9 2 32000 3400
TORCHINDUCTOR_CPP_WRAPPER=1 arm d640_cpp d640 640 9 2 32000 600
GPIC_PROF_DEN_COMPILE_MODE=reduce-overhead arm d640_graphs d640 640 9 2 32000 1200
fail=$required
python $WT/tools/prof/ladder_stages.py $OUT 2>&1 | tee -a $OUT/phases.log
log "DONE fail=$fail $OUT"
exit $fail
