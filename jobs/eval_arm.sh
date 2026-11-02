#!/bin/bash
#SBATCH -J gpic-abl-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# GPIC evaluation of one ablation checkpoint, with the ablation's own code (gpic-wt/abl_run): 50K test captions, the
# official predict (EMA net, 50 Euler steps, CFG 1.0, the official per-prompt noise), all samples in one output.npz,
# then FD-DINOv2 / PRDC / MMD against test_stats.npz with tools/fast_eval.py on all 8 GPUs.
# usage: eval_arm.sh <arm> <step, e.g. 010000>
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
ARM=$1; STEP=$2
WT=/scratch/gpfs/GROUP/USER/project/gpic-wt/abl_run
RUN=$WC/runs/gpic-speed/exp_abl_$ARM
CKPT=$(ls -t $RUN/ms-*$STEP*.ckpt 2>/dev/null | head -1); [ -n "$CKPT" ] || { echo "no ms checkpoint for step $STEP in $RUN"; exit 2; }
OUT=$RUN/eval_$STEP; mkdir -p $OUT; echo "ckpt=$CKPT" | tee $OUT/ckpt.txt
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
t0=$(date +%s)
WANDB_MODE=disabled python main.py predict --config configs_h100/lite_sample.yaml --trainer.devices 8 \
  --trainer.default_root_dir "$OUT/pred" --ckpt_path "$CKPT" > $OUT/predict.log 2>&1
t1=$(date +%s)
NPZ=$(find "$OUT/pred" -name output.npz | head -1)
python -m torch.distributed.run --standalone --nproc_per_node 8 $WT/tools/fast_eval.py "$NPZ" \
  $WC/data/gpic/reference_stats/test_stats.npz $OUT/metrics.json --workers 11 > $OUT/eval.log 2>&1
t2=$(date +%s)
echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-t1))}" | tee $OUT/phase_times.json
cat $OUT/metrics.json
rm -f "$NPZ"   # 50K x 196 KB; metrics.json keeps the result
