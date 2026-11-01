#!/bin/bash
#SBATCH -J gpic-sample-eval
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# GPIC evaluation of a PixelGen checkpoint: 50K test captions -> official predict (8 GPUs, 50 Euler steps, CFG 1.0,
# EMA net, shared seed-0 noise) -> all samples in one output.npz -> FD-DINOv2/PRDC/MMD vs test_stats.npz with
# tools/fast_eval.py (gpic_eval's own model, preprocessing, extractor and metrics, sharded over the 8 GPUs; FD equal to
# the CLI). Phases are timed.
# usage: sample_eval.sh <run_dir with *.ckpt (latest is used) or a .ckpt> <out_dir>
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
SRC=$1; OUT=$2
if [ -d "$SRC" ]; then CKPT=$(ls -t $(find "$SRC" -name "*.ckpt") | head -1); else CKPT=$SRC; fi
mkdir -p "$OUT"; echo "ckpt=$CKPT" | tee $OUT/ckpt.txt
cd $PIXGEN
t0=$(date +%s)
WANDB_MODE=disabled python main.py predict --config configs_h100/lite_sample.yaml --trainer.devices 8 \
  --trainer.default_root_dir "$OUT/pred" --ckpt_path "$CKPT" > $OUT/predict.log 2>&1
t1=$(date +%s)
NPZ=$(find "$OUT/pred" -name output.npz | head -1)
echo "samples: $NPZ"
python -m torch.distributed.run --standalone --nproc_per_node 8 \
  /scratch/gpfs/GROUP/USER/project/gpic-wt/perf-opt/tools/fast_eval.py "$NPZ" \
  /scratch/gpfs/GROUP/USER/data/gpic/reference_stats/test_stats.npz $OUT/metrics.json --workers 11 > $OUT/eval.log 2>&1
t2=$(date +%s)
echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-t1))}" | tee $OUT/phase_times.json
cat $OUT/metrics.json
