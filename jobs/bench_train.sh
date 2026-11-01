#!/bin/bash
# Benchmark run of the official GPIC config (who-ate-my-flops). usage: bench_train.sh <out_dir> [profile a:b]
# Same job as lite_smoke (8 GPUs, batch 32/GPU, bf16, compile); no logger/val/ckpt.
# Env: BENCH_DEVICES (8), BENCH_STEPS (150), BENCH_WRAP (prefix command, e.g. nsys/ncu), BENCH_CUDA_PROFILER, MASTER_PORT.
OUT=$1; PROFILE=${2:-}
mkdir -p "$OUT"
export BENCH_OUT=$OUT BENCH_PROFILE=$PROFILE BENCH_T0=$(date +%s.%N)
export WANDB_MODE=disabled
WT=$(cd "$(dirname "$0")/.." && pwd)
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
git -C $WT rev-parse HEAD > $OUT/commit.txt
env | grep -E "^(BENCH_|TORCH|CUDA_VISIBLE|MASTER_PORT|NCCL)" | sort > $OUT/env.txt
$BENCH_WRAP python main.py fit --config configs_h100/lite_bench.yaml --trainer.devices ${BENCH_DEVICES:-8} \
  --trainer.max_steps ${BENCH_STEPS:-150} --trainer.default_root_dir "$OUT" > "$OUT/stdout.log" 2>&1
rc=$?
echo "rc=$rc" >> "$OUT/stdout.log"
exit $rc
