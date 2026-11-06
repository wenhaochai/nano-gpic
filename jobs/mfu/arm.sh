#!/bin/bash
# One benchmark arm of the MFU work (docs/mfu.md): the ladder's production training path (jobs/train_ladder.sh: config
# configs_h100/lite_ladder.yaml, CompileWarmup, HostTuning, TORCHINDUCTOR_CPP_WRAPPER=1, tcmalloc, region crop, a fixed
# shard seed) with the step timer (src/callbacks/bench.py) and without checkpoints or W&B. Sourced env: jobs/env.sh.
# usage: arm.sh <out dir> <tag> <d> <nb> <nt> <steps> [main.py arguments...]
#   env: MFU_WT (code worktree, default this script's worktree), BENCH_PROFILE / BENCH_STAGES (bench.py), MFU_LIMIT (s,
#   default 1200), anything else the arm changes (NCCL_*, GPIC_* switches), set by the caller.
set -o pipefail
OUT=$1 TAG=$2 D=$3 NB=$4 NT=$5 STEPS=$6; shift 6
WT=${MFU_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
mkdir -p $OUT/$TAG
echo "$(date -u +%FT%TZ) $(git -C $WT rev-parse --short HEAD)$(git -C $WT diff --quiet HEAD -- baselines || echo +dirty) tag=$TAG d=$D nb=$NB nt=$NT steps=$STEPS host=$(hostname) env: $(env | grep -E '^(NCCL_|GPIC_|BENCH_|TORCH|MFU_)' | grep -v -E 'TORCHINDUCTOR_CACHE_DIR|TORCH_HOME' | tr '\n' ' ') args: $*" >> $OUT/arms.txt
CB='[{"class_path":"src.callbacks.compile_warmup.CompileWarmup"},{"class_path":"src.callbacks.host_tuning.HostTuning"},{"class_path":"src.callbacks.bench.BenchTimer"}]'
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD BENCH_OUT=$OUT/$TAG/bench WANDB_MODE=disabled
export GPIC_SHARD_SEED=${GPIC_SHARD_SEED:-1793979619} GPIC_SHARD_LOG_DIR=$OUT/$TAG
export GPIC_REGION_CROP=${GPIC_REGION_CROP:-1} TORCHINDUCTOR_CPP_WRAPPER=${TORCHINDUCTOR_CPP_WRAPPER:-1}
[ -f /usr/lib64/libtcmalloc.so.4 ] && export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4${LD_PRELOAD:+:$LD_PRELOAD} TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=100000000000
# -k: a rank stopped in a warm-up can leave the others in a collective, and SIGTERM does not end them
timeout -k 60 ${MFU_LIMIT:-1200} python main.py fit --config configs_h100/lite_ladder.yaml --trainer.default_root_dir $OUT/$TAG \
  --trainer.max_steps $STEPS --trainer.callbacks "$CB" \
  --model.denoiser.init_args.hidden_size $D --model.denoiser.init_args.num_groups $((D / 64)) \
  --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT \
  --trainer.enable_checkpointing false --tags.exp mfu_$TAG --trainer.logger.init_args.name mfu_$TAG "$@" > $OUT/$TAG/train.log 2>&1
rc=$?
rm -f $OUT/$TAG/exp_mfu_$TAG/*.ckpt
echo "$(date -u +%FT%TZ) tag=$TAG rc=$rc" >> $OUT/arms.txt
exit $rc
