#!/bin/bash
#SBATCH -J gpic-ladder-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#FETCHABLE
# GPIC evaluation of one ladder checkpoint with the ladder's own code (gpic-wt/ladder_run): 50K test captions, the
# official predict (EMA net, 50 Euler steps, CFG 1.0, the official per-prompt noise) with the masked text path, all
# samples in one output.npz, then FD-DINOv2 / PRDC / MMD against test_stats.npz with tools/fast_eval.py on 8 GPUs.
# usage: eval_ladder.sh <size> <d> <nb> <nt> <step>      e.g. eval_ladder.sh d640 640 9 2 129000
# Env (smoke tests only): LADDER_WT, LADDER_RUN_ROOT.
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
SIZE=$1 D=$2 NB=$3 NT=$4 STEP=$(printf %06d $5)
WT=${LADDER_WT:-$WC/project/gpic-wt/ladder_run}
RUN=${LADDER_RUN_ROOT:-$WC/runs/gpic-speed}/exp_ladder_$SIZE
CKPT=$(ls -t $RUN/ms-*$STEP*.ckpt 2>/dev/null | head -1); [ -n "$CKPT" ] || { echo "no ms checkpoint for step $STEP in $RUN"; exit 2; }
OUT=$RUN/eval_$STEP; mkdir -p $OUT; echo "ckpt=$CKPT code=$(git -C $WT rev-parse HEAD)" | tee $OUT/ckpt.txt
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
t0=$(date +%s)
WANDB_MODE=disabled python main.py predict --config configs_h100/lite_ladder_sample.yaml --trainer.devices 8 \
  --trainer.default_root_dir "$OUT/pred" --ckpt_path "$CKPT" \
  --model.denoiser.init_args.hidden_size $D --model.denoiser.init_args.num_groups $((D / 64)) \
  --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT > $OUT/predict.log 2>&1
t1=$(date +%s)
NPZ=$(find "$OUT/pred" -name output.npz | head -1)
python -m torch.distributed.run --standalone --nproc_per_node 8 $WT/tools/fast_eval.py "$NPZ" \
  $WC/data/gpic/reference_stats/test_stats.npz $OUT/metrics.json --workers 11 > $OUT/eval.log 2>&1
t2=$(date +%s)
echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-t1))}" | tee $OUT/phase_times.json
cat $OUT/metrics.json
rm -f "$NPZ"   # 50K x 196 KB; metrics.json keeps the result
