#!/bin/bash
#SBATCH -J gpic-ladder-kernels
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# What is left of the step at the production settings of jobs/train_ladder.sh, on 8 H100s, on one node. Measures only. No W&B.
# 1. d640 and d1280 as production runs them (that combination had no arm in w974). d640 with perf on the main and autograd threads,
#    py-spy and CPU wait.
# 2. torch.profiler traces of both, for the kernel tables (tools/view, tools/prof/roofline.py).
# 3. d640 with the main and autograd threads on their own CPU; with no data path at all (real batches kept in memory), which
#    bounds what any data-path work can give; and both.
# 4. d640 with other kernel layouts, each a cold compile: aggressive fusion, one graph per direction (no DDPOptimizer split),
#    Inductor's memory planning (fewer allocations), GEMM autotuning (Triton kernels launch directly, cuBLAS goes through ATen).
#    Rounding may differ: tools/test_cpp_launcher.py decides before any of these reaches production.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/ladder_kern
[ -d "$WT" ] || { echo "no worktree $WT"; exit 2; }
TS=$(date +%m%d_%H%M)
OUT=$WC/runs/gpic-speed/ladder_kern_$TS
PROF=$WC/tmp/ladder_prof/host_prof/ladder_kern_$TS
mkdir -p $OUT $PROF
git -C $WT rev-parse HEAD > $OUT/commit.txt
export LADDER_WT=$WT LADDER_PARTIAL_OK=1 WANDB_MODE=disabled TORCH_LOGS=recompiles
log() { echo "[$(date +%T)] $*" | tee -a $OUT/phases.log; }

arm() {   # arm <tag> <limit s> <size> <d> <nb> <nt> <steps of this launch> [main.py arguments...]
  local tag=$1 limit=$2 size=$3 d=$4 nb=$5 nt=$6 stop=$7; shift 7
  log "START $tag"
  BENCH_OUT=$OUT/bench/$tag LADDER_RUN_ROOT=$OUT/$tag timeout $limit \
    bash $WT/jobs/train_ladder.sh $size $d $nb $nt 32000 --trainer.max_steps $stop "$@" > $OUT/$tag.log 2>&1
  log "END $tag rc=$?"
}
d640() { local tag=$1 limit=$2 stop=$3; shift 3; arm $tag $limit d640 640 9 2 $stop "$@"; }
d1280() { local tag=$1 limit=$2 stop=$3; shift 3; arm $tag $limit d1280 1280 14 3 $stop "$@"; }

rank_pid() {   # the main process of a rank: LOCAL_RANK in its environment, and a parent that is not the same rank (not a data worker)
  local p r pp rp
  for p in $(pgrep -u $USER -f "main.py fit"); do
    r=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^LOCAL_RANK=//p')
    pp=$(ps -o ppid= -p $p | tr -d ' ')
    rp=$(tr '\0' '\n' < /proc/$pp/environ 2>/dev/null | sed -n 's/^LOCAL_RANK=//p')
    if [ "$r" = "$1" ] && [ "$rp" != "$1" ]; then echo $p; return; fi
  done
}
wait_warm() {   # wait_warm <tag> <seconds after the warm-up>: 1 when the arm ended first
  until grep -q "\[warmup\] done" $OUT/$1.log 2>/dev/null; do
    grep -q "END $1 " $OUT/phases.log && return 1
    sleep 2
  done
  sleep $2
}
perf_thread() {   # perf_thread <tid> <output prefix> <seconds>
  perf record -q -F 999 -t $1 -o $TMPDIR/perf_$$.data -- sleep $3 2>&1 | tail -1
  timeout 200 perf report -i $TMPDIR/perf_$$.data --stdio --no-children --sort dso --percent-limit 0.3 2>/dev/null > $2_dso.txt
  timeout 200 perf report -i $TMPDIR/perf_$$.data --stdio --no-children --sort dso,symbol --percent-limit 0.1 2>/dev/null > $2_symbols.txt
  rm -f $TMPDIR/perf_$$.data
}
cpu_wait() {   # cpu_wait <file> <rank>: CPU use and wait for a CPU of the rank's threads and all python processes over 15 s
  python3 -I $WT/tools/prof/thread_wait.py "$2" 15 > "$1" 2>&1
}
profile_arm() {   # profile_arm <tag> <rank>: perf on the rank's main thread, then its autograd thread, then py-spy and CPU wait
  local tag=$1 rank=$2 dir=$PROF/$1 pid tid
  mkdir -p $dir
  wait_warm $tag 15 || { log "$tag ended before its warm-up was seen"; return; }
  pid=$(rank_pid $rank)
  [ -n "$pid" ] || { log "$tag: no main process for rank $rank"; return; }
  tid=$(grep -l "^pt_autograd_$rank\$" /proc/$pid/task/*/comm 2>/dev/null | head -1 | cut -d/ -f5)
  echo "rank $rank pid $pid autograd tid $tid on $(hostname) $(date +%T)" > $dir/where.txt
  perf_thread $pid $dir/perf_main 15
  [ -n "$tid" ] && perf_thread $tid $dir/perf_autograd 15
  /home/USER/.local/bin/py-spy record --nonblocking -p $pid -d 15 -r 250 -f raw -o $dir/pyspy_raw.txt 2>&1 | tail -1
  cpu_wait $dir/thread_wait.txt $rank
  log "$tag profiled"
}
wait_arm() {   # wait_arm <tag>: thread_wait only, for arms that change where threads run
  mkdir -p $PROF/$1
  wait_warm $1 20 && cpu_wait $PROF/$1/thread_wait.txt 3
}

# 1. Production: about 90 s of profiling on one rank, then steps without it
profile_arm d640_prod 3 &
d640 d640_prod 900 6000
wait
BENCH_PROFILE=1205:1208 d640 d640_trace 600 1300
d1280 d1280_prod 900 1200
BENCH_PROFILE=505:508 d1280 d1280_trace 600 600

# 2. Threads and data path
wait_arm d640_pin &
GPIC_PROF_PIN=1 d640 d640_pin 600 2400
wait
GPIC_PROF_CYCLE_BATCHES=64 d640 d640_cycle 600 2400
wait_arm d640_cycle_pin &
GPIC_PROF_CYCLE_BATCHES=64 GPIC_PROF_PIN=1 d640 d640_cycle_pin 600 2400
wait

# 3. Kernel layouts: cold compiles, the longest last
GPIC_PROF_AGGRESSIVE_FUSION=1 d640 d640_fusion 900 1400
GPIC_PROF_NO_DDP_SPLIT=1 d640 d640_onegraph 900 1400
TORCHINDUCTOR_MEMORY_PLANNING=1 d640 d640_memplan 900 1400
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1 d640 d640_gemm 1500 1400

log "DONE $OUT"
