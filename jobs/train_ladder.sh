#!/bin/bash
#SBATCH -J gpic-ladder
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=08:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable-partial
#FETCHABLE
# One run of the PixelGen ladder's iso-FLOP sweep (docs/ladder_plan.md, sizes and steps in docs/ladder_runs.json):
# configs_h100/lite_ladder.yaml (the official JiT-T2I recipe with grouped, masked text and a flat LR after warmup) at
# width d (heads d/64), nb main blocks and nt text blocks, max_steps steps of 256 images on 8 GPUs.
# Checkpoints: last.ckpt (full state) every max_steps/32 steps for resuming; ms-step=<step>.ckpt (model + EMA) at
# max_steps/4, /2, 3/4 and the end, for the FD evals (jobs/eval_ladder.sh).
# Code: the frozen worktree gpic-wt/ladder_full_run (branch ladder-full, the reviewed commit of the full ladder), so later
# edits cannot reach a running size; the Lite ladder ran gpic-wt/ladder_run.
# usage: train_ladder.sh <size> <d> <nb> <nt> <max_steps> [main.py arguments...]
# Resumable (wq --partial): resumes from the newest last*.ckpt; exits 0 only when max_steps are done. Run again with a
# larger max_steps, it continues the run (flat LR after the warm-up): the full ladder's budget segments (jobs/train_full.sh).
# Env (smoke tests only): LADDER_WT (code worktree), LADDER_RUN_ROOT (run root), BENCH_OUT (adds the step timer),
# LADDER_PARTIAL_OK (a launch stopped early by --trainer.max_steps counts as done). LADDER_LAST_EVERY: the steps between
# two last.ckpt saves (default max_steps/32; jobs/train_full.sh).
# Launch levers (gpic-28 2026-11-09, logs/mfu_infra_scan_1109b.yaml item 6): the same training, less time before the first
# step (128 s a launch at width 1536, 2.8% of the project's GPU-h in 233 launches over 48 h). Each has its own switch, on by
# default; 0 turns one off. jobs/smoke_launch.sh compares both settings.
#  GPIC_LAUNCH_GATE_BG        A1: the GPU health gate runs beside main.py's start; fit waits for its verdict (main.py before_fit)
#  GPIC_LAUNCH_VEC_ISA_MEMO   E: the compiler's CPU vector-set check is answered from a file (src/utils/patch_bugs.py)
#  GPIC_LAUNCH_OMP            F: every rank starts with the thread count Lightning sets at fit (CPUs / ranks), not with all CPUs
#  GPIC_LAUNCH_RANKS          B: this script starts all ranks at once; off by default (1 = on) until jobs/smoke_launch.sh has compared it
#  GPIC_LAUNCH_SKIP_EMA_COPY  G1: with GPIC_MFU_NO_EMA=1, configure_model leaves out the second copy of the weights into the EMA
#                             slot (src/lightning_model_gpic.py); off by default (1 = on) until the same smoke
# GPIC_LAUNCH_FP=1 (tests): src/callbacks/launch_fingerprint.py prints each rank's state at the first step.
# Launch guard (2026-11-10, w2503: wandb.init failed with an HTTP 500 before the first step and the launch hung for 10 min).
# Needs a worktree with src/callbacks/first_step.py (rank 0 writes GPIC_FIRST_STEP_FILE, one file per launch, at the end of
# its first training batch); without it both stay off.
#  GPIC_LAUNCH_RETRIES (2)            a launch that ends with a status other than 0 and 98, with no cut (TERM to this script)
#                                     and no first step, is started again (this script again, by exec: a new gate, the
#                                     newest checkpoint, the shard log cut again) after GPIC_LAUNCH_RETRY_WAIT_S (60) s, at
#                                     most this many times; 0 = off. A cut during the wait: exit 99. After the first step
#                                     nothing changes.
#  GPIC_LAUNCH_START_TIMEOUT_S (1500) a launch that has neither ended nor reached its first step this many seconds after its
#                                     start gets TERM, KILL GPIC_LAUNCH_WATCHDOG_KILL_S (30) s later, and counts as a failure
#                                     before the first step; with no retry left the script ends with 125 (wq: failed, not
#                                     requeued: a launch that hangs 3 times in a row needs a look); 0 = off. Off after a cut.
# Each retry and each watchdog firing is one line in launches.txt. baselines/PixelGen/main.py (GPIC_FAIL_FAST) ends a rank
# whose fit raised within seconds; this script ends the other ranks.
set -o pipefail
# a retry starts this script again with the arguments, directory and environment this run got (the environment of this
# process at its exec, not what the script exported since)
SELF=$(readlink -f "$0") ARGS0=("$@") CWD0=$PWD
mapfile -d '' ENV0 < /proc/$$/environ
ATTEMPT=${GPIC_LAUNCH_ATTEMPT:-0}
export GPIC_LAUNCH_T0=$(date +%s.%N)
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
SIZE=$1 D=$2 NB=$3 NT=$4 STEPS=$5; shift 5
WT=${LADDER_WT:-$WC/project/gpic-wt/ladder_full_run}
ROOT=${LADDER_RUN_ROOT:-$WC/runs/gpic-speed}
RUN=$ROOT/exp_ladder_$SIZE   # main.py: <default_root_dir>/exp_<tags.exp>
if [ $((STEPS % 32)) -ne 0 ] || [ $((D % 64)) -ne 0 ]; then echo "max_steps must be a multiple of 32 and d of 64"; exit 2; fi
MS=$((STEPS / 4)); LAST=${LADDER_LAST_EVERY:-$((STEPS / 32))}   # jobs/train_full.sh sets LADDER_LAST_EVERY per size
[ "$LAST" -gt 0 ] 2>/dev/null || { echo "bad LADDER_LAST_EVERY $LAST"; exit 2; }
mkdir -p $RUN
echo "$(date -u +%FT%TZ) $(git -C $WT rev-parse HEAD) size=$SIZE d=$D nb=$NB nt=$NT max_steps=$STEPS ms_every=$MS last_every=$LAST host=$(hostname) job=$JOBID" >> $RUN/launches.txt
# GPU health gate: a throttled GPU slows the whole run; exit 98 = requeue elsewhere.
# Launch lever A1: the gate in the background, its exit status in a file; main.py starts at once (its first 30 s use no GPU),
# waits for the status before its first CUDA call and exits 98 without a pass. The gate stays in this shell's process group,
# so a cut (TERM to the group) ends it too. Only with a main.py that knows the file.
GATE=$RUN/gpu_health_$(date +%s).txt
GATE_BG=0
[ "${GPIC_LAUNCH_GATE_BG:-1}" = 1 ] && grep -q GPIC_GATE_RC $WT/baselines/PixelGen/main.py && GATE_BG=1
if [ $GATE_BG = 1 ]; then
  export GPIC_GATE_RC=${TMPDIR:-/tmp}/gpu_gate_$$.rc; rm -f $GPIC_GATE_RC
  ( python $WT/tools/gpu_health.py > $GATE 2>&1; echo $? > $GPIC_GATE_RC.tmp; mv $GPIC_GATE_RC.tmp $GPIC_GATE_RC ) &
  export GPIC_GATE_PID=$!
else
  unset GPIC_GATE_RC GPIC_GATE_PID
  python $WT/tools/gpu_health.py > $GATE 2>&1 || { tail -3 $GATE; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
fi
CKPT_ARGS=()
LASTCK=$(ls -t $RUN/last*.ckpt 2>/dev/null | head -1)
if [ -n "$LASTCK" ]; then CKPT_ARGS=(--ckpt_path "$LASTCK"); echo "resuming from $LASTCK"; fi
# Full ladder (docs/ladder_plan.md section 12.4): the run keeps one shard seed (shard_seed.txt: jobs/train_full.sh writes the
# ladder's common seed before the first launch, otherwise the first launch draws one), so every launch draws the same
# permutation in each pass, and the data workers log each shard they start (shards_started_rank<r>.txt; EpochShuffledShards in
# src/data/dataset/gpic_dataset.py), so a resumed launch skips the shards started before its checkpoint and trains no sample
# twice in a pass. The cut is the checkpoint's own dump time (src/callbacks/shard_clock.py, read by tools/ckpt_shard_clock.py;
# the floored mtime only for a checkpoint saved without the clock): a shard started later was read only for steps that this
# launch trains again, so it is read again. A fresh start keeps no line. GPIC_SHARD_SEED from the environment (tests only)
# must match the file. A checkpoint copied without its seed file gets the checkpoint's seed back; one copied without its
# shard log is refused.
SEEDF=$RUN/shard_seed.txt
if [ -n "${GPIC_SHARD_SEED:-}" ]; then
  [ -s $SEEDF ] || echo "$GPIC_SHARD_SEED" > $SEEDF
  [ "$(cat $SEEDF)" = "$GPIC_SHARD_SEED" ] || { echo "GPIC_SHARD_SEED $GPIC_SHARD_SEED differs from $SEEDF ($(cat $SEEDF))"; exit 2; }
fi
KEEP_T=0 CK_SEED=- CK_STEP=0
if [ -n "$LASTCK" ]; then
  read -r KEEP_T CK_SEED CK_STEP < <(python $WT/tools/ckpt_shard_clock.py "$LASTCK") && [ -n "$KEEP_T" ] || { echo "cannot read the shard clock of $LASTCK"; exit 2; }
  if [ "$CK_SEED" != - ]; then
    [ -s $SEEDF ] || echo "$CK_SEED" > $SEEDF
    [ "$(cat $SEEDF)" = "$CK_SEED" ] || { echo "$SEEDF ($(cat $SEEDF)) differs from the seed in $LASTCK ($CK_SEED)"; exit 2; }
  fi
  ls $RUN/shards_started_rank*.txt > /dev/null 2>&1 || { echo "no shard log beside $LASTCK: copy shards_started_rank*.txt with last.ckpt, or the pass reads every shard again"; exit 2; }
fi
[ -s $SEEDF ] || date +%s > $SEEDF
GPIC_SHARD_SEED=$(cat $SEEDF)
for f in $RUN/shards_started_rank*.txt; do
  [ -f "$f" ] || continue
  awk -v T=$KEEP_T '$1 <= T' "$f" > "$f.tmp" && mv "$f.tmp" "$f" || { echo "cannot filter $f"; exit 2; }
done
echo "$(date -u +%FT%TZ) shard seed $GPIC_SHARD_SEED, shard log kept up to $KEEP_T (checkpoint step $CK_STEP): $(cat $RUN/shards_started_rank*.txt 2>/dev/null | wc -l) lines" >> $RUN/launches.txt
BENCH_CB=""
# launch guard (top): the first-step marker, when this worktree has it
FIRST_STEP=0
grep -q "class FirstStepMarker" $WT/baselines/PixelGen/src/callbacks/first_step.py 2> /dev/null && FIRST_STEP=1
[ $FIRST_STEP = 1 ] && BENCH_CB=',{"class_path":"src.callbacks.first_step.FirstStepMarker"}'
if [ -n "${BENCH_OUT:-}" ]; then BENCH_CB=$BENCH_CB',{"class_path":"src.callbacks.bench.BenchTimer"}'; fi
# tests only, the last callback: it prints the state the first step starts from and changes nothing
if [ "${GPIC_LAUNCH_FP:-0}" = 1 ]; then BENCH_CB=$BENCH_CB',{"class_path":"src.callbacks.launch_fingerprint.LaunchFingerprint"}'; fi
# the ms hook first: a kill between the two saves must not leave last.ckpt ahead of a missing ms file (review N1);
# the last hook without a version counter: last.ckpt is always the live state, also in a branch or an extension
# (src/callbacks/model_checkpoint.py); the ms hook keeps its counter, so no ms file is ever written over
CB='[{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"ms-{step:06d}","monitor":null,"save_last":false,"save_top_k":-1,"every_n_train_steps":'$MS'}},{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"resume-{step:06d}","monitor":null,"save_last":true,"save_top_k":0,"every_n_train_steps":'$LAST',"enable_version_counter":false}},{"class_path":"src.callbacks.compile_warmup.CompileWarmup"},{"class_path":"src.callbacks.host_tuning.HostTuning"},{"class_path":"src.callbacks.shard_clock.ShardClock"},{"class_path":"src.callbacks.save_on_term.SaveOnTerm","init_args":{"dirpath":"'$RUN'"}},{"class_path":"src.callbacks.none_refcount.NoneRefcount","init_args":{"path":"'$RUN/none_refcount.txt'","every":1000}}'$BENCH_CB']'
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
export GPIC_SHARD_SEED GPIC_SHARD_LOG_DIR=$RUN   # one shard order per run and pass, shared by all ranks (above)
export GPIC_REGION_CROP=${GPIC_REGION_CROP:-1}   # the crop resizes only the region it keeps: d640 29.5 -> 26.7 ms (w1117); the same random draws; 0.035% of the channel values differ, by at most 2 of 255
export TORCHINDUCTOR_CPP_WRAPPER=${PROBE_CPP_WRAPPER:-1}   # kernels launched from C++: d640 39.3 -> 32.6 ms; the same math up to bf16 rounding (tools/test_cpp_launcher.py)
# MFU levers (docs/mfu.md, branch mfu): the same math, each behind its own switch, on by default; set one to 0 (A: sdpa) to
# turn it off (E is off by default). A: FlashAttention-3 for the masked attention of the image blocks. B: FSDP2 per block in place of DDP
# (bf16 compute parameters, fp32 reduce-scatter, sharded AdamW and EMA; ModelParallelStrategy below). C: the region crop's
# resize on the GPU, bit for bit Pillow's. D: uint8 batches, mapped to the transform's floats on the GPU. E: packed captions,
# the samples of each step dealt over the ranks by caption length. F: checkpoint writes in a separate process.
export GPIC_MFU_ATTN=${GPIC_MFU_ATTN:-fa3}
# B helps where the step is GPU-bound and hurts where the host binds it (production path, 8 GPUs, w1389/w1390: d1280 76.0 ->
# 64.9 ms with C + D; d640 26.6 -> 43.9 ms): on from d1280 by default (d1024: measured in w139x)
export GPIC_MFU_FSDP=${GPIC_MFU_FSDP:-$([ $D -ge 1280 ] && echo 1 || echo 0)}
export GPIC_MFU_GPU_RESIZE=${GPIC_MFU_GPU_RESIZE:-1} GPIC_MFU_UINT8=${GPIC_MFU_UINT8:-1}
export GPIC_MFU_PACK=${GPIC_MFU_PACK:-0} GPIC_MFU_REBALANCE=${GPIC_MFU_REBALANCE:-0}   # E: off until its slowdown is fixed (docs/mfu.md)
# E2: GPIC_MFU_PACK=1 GPIC_MFU_REBALANCE=2 deals the samples on the host one step ahead (src/data/rebalance_loader.py)
GPIC_MFU_ASYNC_CKPT=${GPIC_MFU_ASYNC_CKPT:-1}
# B2 (AdamW and EMA on the shards' local tensors) and FSDP2 blocks per unit: under test (docs/mfu.md), off by default
export GPIC_MFU_LOCAL_OPT=${GPIC_MFU_LOCAL_OPT:-0} GPIC_MFU_FSDP_GROUP=${GPIC_MFU_FSDP_GROUP:-1}
export GPIC_MFU_FUSED_OPT=${GPIC_MFU_FUSED_OPT:-0}   # MFU lever H: 1 = clip multiply + AdamW in one kernel, 2 = also the EMA
export GPIC_MFU_FUSED_NORM=${GPIC_MFU_FUSED_NORM:-0}   # MFU lever K: residual + RMSNorm + modulation as Triton kernels
export GPIC_MFU_FUSED_QK=${GPIC_MFU_FUSED_QK:-0}       # MFU lever K2: q/k norm + RoPE + k/v concatenation as Triton kernels
export GPIC_MFU_FP8=${GPIC_MFU_FP8:-}   # MFU lever F8: tensorwise | rowwise FP8 GEMMs in the blocks (changes the numerics; empty = bf16)
MFU_ARGS=()
export GPIC_MFU_FSDP_FLAT=${GPIC_MFU_FSDP_FLAT:-0}   # MFU lever B3: one flat parameter per FSDP2 unit (src/utils/fsdp_setup.py)
FSDP_CLS=lightning.pytorch.strategies.ModelParallelStrategy
[ "$GPIC_MFU_FSDP_FLAT" = 1 ] && FSDP_CLS=src.utils.fsdp_setup.FlatModelParallelStrategy   # checkpoints keep the per-parameter keys
# new ladders (global batch 1024 by accumulation): no FSDP2 reduce-scatter or reshard before the last micro-batch
[ "${GPIC_MFU_FSDP_NOSYNC:-0}" = 1 ] && FSDP_CLS=src.utils.fsdp_setup.NoSyncModelParallelStrategy
[ "$GPIC_MFU_FSDP" = 1 ] && MFU_ARGS+=(--trainer.strategy '{"class_path":"'$FSDP_CLS'","init_args":{"data_parallel_size":8,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}')
[ "$GPIC_MFU_ASYNC_CKPT" = 1 ] && MFU_ARGS+=(--trainer.plugins+='{"class_path":"src.plugins.async_ckpt_io.AsyncShmCheckpointIO"}')
echo "$(date -u +%FT%TZ) MFU levers: attn=$GPIC_MFU_ATTN fsdp=$GPIC_MFU_FSDP gpu_resize=$GPIC_MFU_GPU_RESIZE uint8=$GPIC_MFU_UINT8 pack=$GPIC_MFU_PACK rebalance=$GPIC_MFU_REBALANCE async_ckpt=$GPIC_MFU_ASYNC_CKPT local_opt=$GPIC_MFU_LOCAL_OPT fsdp_group=$GPIC_MFU_FSDP_GROUP fsdp_flat=$GPIC_MFU_FSDP_FLAT fused_opt=$GPIC_MFU_FUSED_OPT fused_norm=$GPIC_MFU_FUSED_NORM fused_qk=$GPIC_MFU_FUSED_QK fp8=${GPIC_MFU_FP8:-none} fsdp_nosync=${GPIC_MFU_FSDP_NOSYNC:-0}" >> $RUN/launches.txt
# the main thread spent 14.5% of its CPU time in glibc malloc; the threshold keeps tcmalloc's large-alloc report out of the log
[ -f /usr/lib64/libtcmalloc.so.4 ] && export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4${LD_PRELOAD:+:$LD_PRELOAD} TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=100000000000
# the pilot's cut (fetch_pending: timeout TERMs the whole process group, KILL 300 s later): this shell waits for main.py, which
# saves at the next step all ranks agree on and stops (src/callbacks/save_on_term.py). Without the trap the TERM ended this shell
# at once, its EXIT trap (env.sh) removed TMPDIR, and every rank's pin-memory thread failed on the DataLoader's sockets in it
# before any save (smoke w2409); timeout also saw its child gone and freed the GPUs while main.py still ran.
CUT=0   # set by the TERM: no retry and no watchdog after a cut
trap 'CUT=1; echo "$(date -u +%FT%TZ) train_ladder.sh: TERM; main.py saves and stops"' TERM
# Launch lever E: a True answer of the vector-set check is kept here, one empty file per compiler, flags, torch build and CPU
if [ "${GPIC_LAUNCH_VEC_ISA_MEMO:-1}" = 1 ]; then export GPIC_VEC_ISA_MEMO=${GPIC_VEC_ISA_MEMO:-$WC/cache/inductor/vec_isa_memo}; else unset GPIC_VEC_ISA_MEMO; fi
# Launch lever F: Lightning gives every rank CPUs / ranks threads when fit starts (_set_num_threads_if_needed), but until then
# each rank builds the model with one thread per CPU, 7 x 96 threads on 96 CPUs while ranks 1-7 start (their model
# construction took 17.8 s against 11.0 s for rank 0 alone, w2441). The same count from the start; a count set by the caller stays.
NDEV=8 prev=""   # the number of ranks: configs_h100/lite_ladder.yaml has devices '8'; the caller's --trainer.devices wins
for a in "$@"; do
  [ "$prev" = --trainer.devices ] && NDEV=$a
  case $a in --trainer.devices=*) NDEV=${a#*=} ;; esac
  prev=$a
done
[ "$NDEV" -gt 0 ] 2> /dev/null || NDEV=0   # not a count (auto, a list of GPUs): levers F and B stay off
OMP_ENV=()
if [ "${GPIC_LAUNCH_OMP:-1}" = 1 ] && [ -z "${OMP_NUM_THREADS:-}" ] && [ $NDEV -gt 0 ]; then
  NTHR=$(( $(nproc) / NDEV )); [ $NTHR -ge 1 ] || NTHR=1
  OMP_ENV=(env OMP_NUM_THREADS=$NTHR)
fi
# Launch lever B: Lightning's rank 0 imports, builds the model and only then, in fit, starts ranks 1-7 with its own command
# line; they import and build again while rank 0 waits (27.8 s of the 128 s before the first step at width 1536, w2441).
# Here this script starts every rank with that command line at once. LOCAL_RANK in the environment tells Lightning that
# the ranks exist (LightningEnvironment.creates_processes_externally), and MASTER_ADDR, MASTER_PORT, NODE_RANK and WORLD_SIZE
# are the variables its launcher gives the ranks it starts; rank 0 then runs as ranks 1-7 always did. Lightning's watcher
# of the ranks (a rank that ends with an error: KILL for all) moves into this script, which also waits for every rank
# (before: the script ended with rank 0, and the pilot found ranks 1-7 still at their exit in 7% of the launches).
RANKS_BG=0
if [ "${GPIC_LAUNCH_RANKS:-0}" = 1 ] && [ $NDEV -gt 1 ]; then
  # a port that is free now, below the range the system hands out for outgoing connections (32768 and up), and chosen from this
  # shell's id: rank 0 binds it some 20 s from now, and two launches on one node (4 GPUs each) must not pick the same one
  RANK_PORT=$(python - $$ <<'PY'
import socket, sys
for k in range(200):
    port = 10000 + (int(sys.argv[1]) * 7919 + k * 101) % 20000
    with socket.socket() as s:
        try:
            s.bind(("", port))   # every interface, as the store of rank 0 binds it
        except OSError:
            continue
    print(port)
    break
PY
)
  [ -n "$RANK_PORT" ] && RANKS_BG=1   # no free port among 200: Lightning starts the ranks as before
fi
# Launch lever G1: in force only with no EMA and a model file that knows the switch
SKIP_EMA=0
[ "${GPIC_LAUNCH_SKIP_EMA_COPY:-0}" = 1 ] && [ "${GPIC_MFU_NO_EMA:-0}" = 1 ] && grep -q GPIC_LAUNCH_SKIP_EMA_COPY $WT/baselines/PixelGen/src/lightning_model_gpic.py 2> /dev/null && SKIP_EMA=1
echo "$(date -u +%FT%TZ) launch levers: gate_bg=$GATE_BG vec_isa_memo=${GPIC_VEC_ISA_MEMO:-off} omp=${OMP_ENV[1]:-off} ranks=$RANKS_BG skip_ema_copy=$SKIP_EMA" >> $RUN/launches.txt
# Every process of this launch carries this name in its environment (ranks, data workers, the checkpoint writer): a rank
# that fails must leave none of them behind (below)
export GPIC_LAUNCH_ID=$$-$(date +%s%N)
launch_left() {
  python - "$GPIC_LAUNCH_ID" $$ <<'PY'
import os, sys
key, skip = ("GPIC_LAUNCH_ID=" + sys.argv[1]).encode(), {int(sys.argv[2]), os.getpid(), os.getppid()}
for p in os.listdir("/proc"):
    if p.isdigit() and int(p) not in skip:
        try:
            if os.stat(f"/proc/{p}").st_uid == os.getuid() and key in open(f"/proc/{p}/environ", "rb").read().split(b"\0"):
                print(p)
        except OSError:   # a process that ended, or another user's
            pass
PY
}
# Launch guard (top of this file): the marker of this launch's first step, the retry count and the start watchdog
RETRIES=${GPIC_LAUNCH_RETRIES:-2} START_TO=${GPIC_LAUNCH_START_TIMEOUT_S:-1500}
[[ $RETRIES =~ ^[0-9]+$ ]] || RETRIES=0
[[ $START_TO =~ ^[0-9]+$ ]] || START_TO=0
[ $FIRST_STEP = 1 ] || { RETRIES=0 START_TO=0; }
if [ $FIRST_STEP = 1 ]; then export GPIC_FIRST_STEP_FILE=${TMPDIR:-/tmp}/first_step_$GPIC_LAUNCH_ID; rm -f $GPIC_FIRST_STEP_FILE; else unset GPIC_FIRST_STEP_FILE; fi
echo "$(date -u +%FT%TZ) launch guard: attempt=$ATTEMPT retries=$RETRIES retry_wait=${GPIC_LAUNCH_RETRY_WAIT_S:-60} start_timeout=$START_TO" >> $RUN/launches.txt
WD_FIRED=0
# true when the watchdog must end this launch now: on, no cut, no first step yet, the time is up
watchdog_due() {
  [ $START_TO -gt 0 ] && [ $CUT = 0 ] && [ $WD_FIRED = 0 ] && [ ! -e "$GPIC_FIRST_STEP_FILE" ] && [ $((SECONDS - LAUNCH_S)) -ge $START_TO ]
}
# the watchdog's action: TERM to the given processes, then KILL for every process of this launch after
# GPIC_LAUNCH_WATCHDOG_KILL_S (30) s, or at once when they have all ended
watchdog_fire() {
  WD_FIRED=1
  local msg="launch watchdog: no first step and no exit $((SECONDS - LAUNCH_S)) s after the start (limit $START_TO s); TERM, KILL after ${GPIC_LAUNCH_WATCHDOG_KILL_S:-30} s"
  echo "$(date -u +%FT%TZ) train_ladder.sh: $msg"
  echo "$(date -u +%FT%TZ) $msg" >> $RUN/launches.txt
  kill -TERM "$@" 2> /dev/null
  local lim=$((SECONDS + ${GPIC_LAUNCH_WATCHDOG_KILL_S:-30})) p alive
  while [ $SECONDS -lt $lim ]; do
    alive=0; for p in "$@"; do kill -0 $p 2> /dev/null && alive=1; done
    [ $alive = 1 ] || break
    sleep 0.5
  done
  local lp=$(launch_left)
  [ -n "$lp" ] && kill -KILL $lp 2> /dev/null
}
LAUNCH_S=$SECONDS
CMD=(python main.py fit --config configs_h100/lite_ladder.yaml --trainer.default_root_dir $ROOT
  --trainer.max_steps $STEPS --trainer.callbacks "$CB"
  --model.denoiser.init_args.hidden_size $D --model.denoiser.init_args.num_groups $((D / 64))
  --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT
  --tags.exp ladder_$SIZE --trainer.logger.init_args.name ladder_$SIZE --trainer.logger.init_args.id ladder-$SIZE
  "${MFU_ARGS[@]}" "$@" "${CKPT_ARGS[@]}")
if [ $RANKS_BG = 0 ]; then
  if [ $START_TO = 0 ]; then
    "${OMP_ENV[@]}" "${CMD[@]}"
    rc=$?
  else   # in the background, so the watchdog can look every 0.5 s; the TERM trap runs at once, main.py gets the cut's TERM itself
    # (after the first step or a cut it only waits)
    "${OMP_ENV[@]}" "${CMD[@]}" & MPID=$!
    while kill -0 $MPID 2> /dev/null && [ $CUT = 0 ] && [ ! -e "$GPIC_FIRST_STEP_FILE" ]; do
      if watchdog_due; then watchdog_fire $MPID; break; fi
      sleep 0.5
    done
    while kill -0 $MPID 2> /dev/null; do wait $MPID; done   # a TERM ends a wait early: wait again
    wait $MPID; rc=$?   # bash keeps the status of an ended child: this is main.py's own
    [ $WD_FIRED = 1 ] && rc=125
  fi
else
  export MASTER_ADDR=127.0.0.1 MASTER_PORT=$RANK_PORT NODE_RANK=0 WORLD_SIZE=$NDEV
  RPIDS=()
  for r in $(seq 0 $((NDEV - 1))); do LOCAL_RANK=$r "${OMP_ENV[@]}" "${CMD[@]}" & RPIDS+=($!); done   # rank 0 first: the oldest process
  LEFT=("${RPIDS[@]}"); rc=0; R0OK=0; TICK=""
  # Part 1, until rank 0 ends with 0 or a rank fails: wait for the next rank that ends. While the start watchdog is armed, a
  # 5-s sleep beside the ranks wakes the wait, and the watchdog looks.
  while [ ${#LEFT[@]} -gt 0 ]; do
    if [ -n "$TICK" ] && ! kill -0 $TICK 2> /dev/null; then wait $TICK 2> /dev/null; TICK=""; fi
    if watchdog_due; then
      watchdog_fire "${LEFT[@]}"
      for p in "${LEFT[@]}"; do wait $p 2> /dev/null; done
      LEFT=(); rc=125; break
    fi
    if [ -z "$TICK" ] && [ $START_TO -gt 0 ] && [ $CUT = 0 ] && [ ! -e "$GPIC_FIRST_STEP_FILE" ]; then sleep 5 & TICK=$!; fi
    dp=""; wait -n -p dp "${LEFT[@]}" $TICK; st=$?
    if [ -n "$TICK" ] && [ "$dp" = "$TICK" ]; then TICK=""; continue; fi
    if [ -z "$dp" ]; then   # the TERM trap ended the wait (the ranks save and stop), or a rank ended that wait -n no longer reports
      for p in "${LEFT[@]}"; do kill -0 $p 2> /dev/null || { dp=$p; wait $p; st=$?; break; }; done
      [ -n "$dp" ] || continue
    fi
    NEW=(); for p in "${LEFT[@]}"; do [ $p = $dp ] || NEW+=($p); done; LEFT=("${NEW[@]}")
    if [ $st -ne 0 ]; then   # the first rank that fails gives the launch its status
      rc=$st
      # 98 is the gate's verdict (main.py before_fit): every rank reads it and ends by itself. After any other failure the
      # other ranks would wait for this one in a collective: KILL, as Lightning's watcher did
      if [ $st -ne 98 ] && [ ${#LEFT[@]} -gt 0 ]; then echo "train_ladder.sh: rank $(for i in "${!RPIDS[@]}"; do [ ${RPIDS[$i]} = $dp ] && echo $i; done) ended with $st; KILL for the other ${#LEFT[@]}"; kill -KILL "${LEFT[@]}" 2> /dev/null; fi
      break
    elif [ $dp = ${RPIDS[0]} ]; then R0OK=1; break
    fi
  done
  [ -n "$TICK" ] && { kill $TICK 2> /dev/null; wait $TICK 2> /dev/null; TICK=""; }
  # Part 2: the ranks that are left end by themselves, within GPIC_LAUNCH_RANKS_EXIT_S (120) seconds, or get KILL: no rank keeps
  # the GPUs. After rank 0 ended with 0 (its saves are done) what a later rank does at its exit does not change the launch.
  # A look every 0.2 s and no helper process: a helper that must be stopped can be left behind (a TERM between its fork and its
  # first command is lost; tools/mfu/tests/test_launch_ranks.py found one in 1 of 8 launches).
  LIMIT=$(( SECONDS + ${GPIC_LAUNCH_RANKS_EXIT_S:-120} ))
  while [ ${#LEFT[@]} -gt 0 ]; do
    NEW=()
    for p in "${LEFT[@]}"; do
      if kill -0 $p 2> /dev/null; then NEW+=($p)
      else wait $p; st=$?; [ $st -eq 0 ] || [ $R0OK = 0 ] || echo "train_ladder.sh: a rank ended with $st after rank 0 ended with 0"; fi
    done
    LEFT=("${NEW[@]}")
    [ ${#LEFT[@]} -gt 0 ] || break
    if [ $SECONDS -ge $LIMIT ]; then echo "train_ladder.sh: ${#LEFT[@]} ranks did not end in ${GPIC_LAUNCH_RANKS_EXIT_S:-120} s; KILL"; kill -KILL "${LEFT[@]}" 2> /dev/null; LIMIT=$(( SECONDS + 3600 )); fi
    sleep 0.2
  done
fi
if [ $GATE_BG = 1 ]; then
  if [ ! -e $GPIC_GATE_RC ]; then   # main.py ended before the verdict: stop the gate and its per-GPU workers
    GATE_PY=$(pgrep -P $GPIC_GATE_PID | paste -sd,)
    GATE_WK=$([ -n "$GATE_PY" ] && pgrep -P $GATE_PY | paste -sd' ')
    pkill -P $GPIC_GATE_PID 2> /dev/null
    [ -n "$GATE_WK" ] && kill $GATE_WK 2> /dev/null
  fi
  wait $GPIC_GATE_PID 2> /dev/null
  rm -f $GPIC_GATE_RC $GPIC_GATE_RC.tmp
  if [ $rc -eq 98 ]; then tail -3 $GATE; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; fi
fi
# A launch that failed leaves no process. Lightning ends ranks 1-7 when rank 0 fails, but not when its own teardown fails
# too: in w2514 rank 0 left with 97 (the checkpoint writer had died) and ranks 1-7 and their data workers, which ignore TERM,
# stayed on the GPUs until the pilot's sweep; with lever B the ranks get KILL above, their data workers do not. So after a
# status other than 0: what still carries this launch's name 5 s later gets KILL.
if [ $rc -ne 0 ]; then
  LEFT_P=$(launch_left)
  if [ -n "$LEFT_P" ]; then sleep 5; LEFT_P=$(launch_left); fi
  if [ -n "$LEFT_P" ]; then
    echo "train_ladder.sh: the launch ended with $rc and $(echo $LEFT_P | wc -w) of its processes are still there; KILL"
    kill -KILL $LEFT_P 2> /dev/null
  fi
fi
# Launch guard: a failure before the first step of this launch (no marker file), not a cut, not the gate's 98: wait, then this
# script again with the same arguments (exec: the same process, so wq and the pilot see one job), up to RETRIES times
STARTED=0; [ -n "${GPIC_FIRST_STEP_FILE:-}" ] && [ -e "$GPIC_FIRST_STEP_FILE" ] && STARTED=1
rm -f "${GPIC_FIRST_STEP_FILE:-}" "${GPIC_FIRST_STEP_FILE:-}.tmp"
if [ $rc -ne 0 ] && [ $rc -ne 98 ] && [ $CUT = 0 ] && [ $STARTED = 0 ] && [ $FIRST_STEP = 1 ] && [ $RETRIES -gt 0 ]; then
  WHY=$([ $WD_FIRED = 1 ] && echo "watchdog: no first step in $START_TO s" || echo "failed before the first step")
  if [ $ATTEMPT -lt $RETRIES ]; then
    N=$((ATTEMPT + 1)) WAIT_S=${GPIC_LAUNCH_RETRY_WAIT_S:-60}
    echo "$(date -u +%FT%TZ) launch retry $N: status $rc, reason $WHY; next launch in $WAIT_S s" >> $RUN/launches.txt
    echo "$(date -u +%FT%TZ) train_ladder.sh: launch retry $N of $RETRIES: status $rc ($WHY); next launch in $WAIT_S s"
    sleep $WAIT_S & SP=$!
    wait $SP   # a TERM (the pilot's cut) ends the wait at once
    kill $SP 2> /dev/null; wait $SP 2> /dev/null
    if [ $CUT = 1 ]; then
      echo "$(date -u +%FT%TZ) launch retry $N: cut during the wait; exit 99" >> $RUN/launches.txt
      echo "main.py exit 99 (a cut during the wait before launch retry $N)"; exit 99
    fi
    # the same process (wq and the pilot see one job) with the first environment; env.sh sets TMPDIR's EXIT trap again
    cd "$CWD0"
    exec env -i "${ENV0[@]}" GPIC_LAUNCH_ATTEMPT=$N bash "$SELF" "${ARGS0[@]}"
  fi
  echo "$(date -u +%FT%TZ) launch retry: none left ($RETRIES used); status $rc, reason $WHY" >> $RUN/launches.txt
elif [ $rc -ne 0 ] && [ $rc -ne 98 ] && [ $rc -ne 99 ] && [ $rc -ne 124 ] && [ $rc -ne 143 ] && [ $CUT = 1 ] && [ $STARTED = 0 ] && [ $FIRST_STEP = 1 ] && [ $ATTEMPT -lt $RETRIES ]; then
  # a cut that came while a launch failed before its first step: nothing to save, and a retry was still left: requeue
  echo "$(date -u +%FT%TZ) launch retry: cut after a failure (status $rc) before the first step; exit 99" >> $RUN/launches.txt
  rc=99
fi
# Lightning turns a SIGTERM that reaches only the ranks into exit 0: done means the final ms checkpoint exists (review S1)
if [ $rc -eq 0 ] && [ -z "${LADDER_PARTIAL_OK:-}" ] && ! ls $RUN/ms-step=$(printf %06d $STEPS)*.ckpt > /dev/null 2>&1; then
  echo "main.py exit 0 before max_steps $STEPS"; rc=99
fi
echo "main.py exit $rc"
exit $rc
