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
set -o pipefail
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
# GPU health gate: a throttled GPU slows the whole run; exit 98 = requeue elsewhere
python $WT/tools/gpu_health.py > $RUN/gpu_health_$(date +%s).txt 2>&1 || { tail -3 $(ls -t $RUN/gpu_health_*.txt | head -1); echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
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
if [ -n "${BENCH_OUT:-}" ]; then BENCH_CB=',{"class_path":"src.callbacks.bench.BenchTimer"}'; fi
# the ms hook first: a kill between the two saves must not leave last.ckpt ahead of a missing ms file (review N1)
CB='[{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"ms-{step:06d}","monitor":null,"save_last":false,"save_top_k":-1,"every_n_train_steps":'$MS'}},{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"resume-{step:06d}","monitor":null,"save_last":true,"save_top_k":0,"every_n_train_steps":'$LAST'}},{"class_path":"src.callbacks.compile_warmup.CompileWarmup"},{"class_path":"src.callbacks.host_tuning.HostTuning"},{"class_path":"src.callbacks.shard_clock.ShardClock"}'$BENCH_CB']'
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
export GPIC_SHARD_SEED GPIC_SHARD_LOG_DIR=$RUN   # one shard order per run and pass, shared by all ranks (above)
export GPIC_REGION_CROP=${GPIC_REGION_CROP:-1}   # the crop resizes only the region it keeps: d640 29.5 -> 26.7 ms (w1117); the same random draws; 0.035% of the channel values differ, by at most 2 of 255
export TORCHINDUCTOR_CPP_WRAPPER=1   # kernels launched from C++: d640 39.3 -> 32.6 ms; the same math up to bf16 rounding (tools/test_cpp_launcher.py)
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
GPIC_MFU_ASYNC_CKPT=${GPIC_MFU_ASYNC_CKPT:-1}
# B2 (AdamW and EMA on the shards' local tensors) and FSDP2 blocks per unit: under test (docs/mfu.md), off by default
export GPIC_MFU_LOCAL_OPT=${GPIC_MFU_LOCAL_OPT:-0} GPIC_MFU_FSDP_GROUP=${GPIC_MFU_FSDP_GROUP:-1}
MFU_ARGS=()
[ "$GPIC_MFU_FSDP" = 1 ] && MFU_ARGS+=(--trainer.strategy '{"class_path":"lightning.pytorch.strategies.ModelParallelStrategy","init_args":{"data_parallel_size":8,"tensor_parallel_size":1,"save_distributed_checkpoint":false}}')
[ "$GPIC_MFU_ASYNC_CKPT" = 1 ] && MFU_ARGS+=(--trainer.plugins+='{"class_path":"src.plugins.async_ckpt_io.AsyncShmCheckpointIO"}')
echo "$(date -u +%FT%TZ) MFU levers: attn=$GPIC_MFU_ATTN fsdp=$GPIC_MFU_FSDP gpu_resize=$GPIC_MFU_GPU_RESIZE uint8=$GPIC_MFU_UINT8 pack=$GPIC_MFU_PACK rebalance=$GPIC_MFU_REBALANCE async_ckpt=$GPIC_MFU_ASYNC_CKPT local_opt=$GPIC_MFU_LOCAL_OPT fsdp_group=$GPIC_MFU_FSDP_GROUP" >> $RUN/launches.txt
# the main thread spent 14.5% of its CPU time in glibc malloc; the threshold keeps tcmalloc's large-alloc report out of the log
[ -f /usr/lib64/libtcmalloc.so.4 ] && export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4${LD_PRELOAD:+:$LD_PRELOAD} TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=100000000000
python main.py fit --config configs_h100/lite_ladder.yaml --trainer.default_root_dir $ROOT \
  --trainer.max_steps $STEPS --trainer.callbacks "$CB" \
  --model.denoiser.init_args.hidden_size $D --model.denoiser.init_args.num_groups $((D / 64)) \
  --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT \
  --tags.exp ladder_$SIZE --trainer.logger.init_args.name ladder_$SIZE --trainer.logger.init_args.id ladder-$SIZE \
  "${MFU_ARGS[@]}" "$@" "${CKPT_ARGS[@]}"
rc=$?
# Lightning turns a SIGTERM that reaches only the ranks into exit 0: done means the final ms checkpoint exists (review S1)
if [ $rc -eq 0 ] && [ -z "${LADDER_PARTIAL_OK:-}" ] && ! ls $RUN/ms-step=$(printf %06d $STEPS)*.ckpt > /dev/null 2>&1; then
  echo "main.py exit 0 before max_steps $STEPS"; rc=99
fi
echo "main.py exit $rc"
exit $rc
