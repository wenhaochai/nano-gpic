#!/bin/bash
#SBATCH -J gpic-ar-eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GPIC evaluation of one autoregressive checkpoint: jobs/eval_ladder.sh's protocol (50K test captions, EMA net, CFG 1.0, FD-DINOv2 / PRDC / MMD), with configs_h100/ar_sample.yaml. x_T is not used: each image's randomness comes from the draws. So the baseline is the point model with each prompt's own noise.
# usage: eval_ar.sh <size> <d> <nb> <nt> <step> [base|first<K>|eta<E>]   The last is the sampler: as drawn, the first K tokens drawn again at the end, or the jump process with stochasticity E. All three give each prompt its own noise.
# Env (smoke tests only): LADDER_WT, LADDER_RUN_ROOT.
set -eo pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
NAME=$1 D=$2 NB=$3 NT=$4 STEP=$(printf %06d $5) VARIANT=${6:-base}
case $VARIANT in
  base) SAMPLER= ;;
  first*) SAMPLER="--model.denoiser.init_args.redraw_first ${VARIANT#first}" ;;
  eta*) SAMPLER="--model.denoiser.init_args.eta ${VARIANT#eta}" ;;
  *) echo "unknown variant $VARIANT"; exit 2 ;;
esac
WT=${LADDER_WT:-$WC/project/gpic-wt/ar_run}
RUN=${LADDER_RUN_ROOT:-$WC/runs/gpic-speed}/exp_ladder_$NAME
CKPT=$(ls -t $RUN/ms-*$STEP*.ckpt 2>/dev/null | head -1); [ -n "$CKPT" ] || { echo "no ms checkpoint for step $STEP in $RUN"; exit 2; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD
python - "$CKPT" $D $NB $NT <<'PY'
import re, sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)["state_dict"]
d, nb, nt = map(int, sys.argv[2:])
n = lambda part: len({m.group(1) for k in sd for m in [re.match(rf"ema_denoiser\.{part}\.(\d+)\.", k)] if m})
got = (sd["ema_denoiser.y_pos_embedding"].shape[-1], n("blocks"), n("text_refine_blocks"))
assert got == (d, nb, nt), f"checkpoint has d, nb, nt = {got}, arguments {(d, nb, nt)}"
assert "ema_denoiser.head.out.weight" in sd, "this checkpoint is not the autoregressive model"
PY
OUT=$RUN/eval_${STEP}_ar_$VARIANT; mkdir -p $OUT; echo "ckpt=$CKPT code=$(git -C $WT rev-parse HEAD)" | tee $OUT/ckpt.txt
t0=$(date +%s)
WANDB_MODE=disabled python main.py predict --config configs_h100/lite_ladder_sample.yaml --config configs_h100/ar_sample.yaml --trainer.devices 8 \
  --trainer.default_root_dir "$OUT/pred" --ckpt_path "$CKPT" \
  --model.denoiser.init_args.hidden_size $D --model.denoiser.init_args.num_groups $((D / 64)) \
  --model.denoiser.init_args.num_blocks $NB --model.denoiser.init_args.num_text_blocks $NT $SAMPLER > $OUT/predict.log 2>&1
t1=$(date +%s)
NPZ=$(find "$OUT/pred" -name output.npz | head -1)
python -m torch.distributed.run --standalone --nproc_per_node 8 $WT/tools/fast_eval.py "$NPZ" $WC/data/gpic/reference_stats/test_stats.npz $OUT/metrics.json --workers 11 > $OUT/eval.log 2>&1
t2=$(date +%s)
echo "{\"predict_s\": $((t1-t0)), \"eval_s\": $((t2-t1))}" | tee $OUT/phase_times.json
cat $OUT/metrics.json
python - "$NPZ" $OUT/first.npy <<'PY'
import sys
import numpy as np
np.save(sys.argv[2], np.load(sys.argv[1])["arr_0"][:64])
PY
rm -f "$NPZ"   # 50K x 196 KB; metrics.json keeps the result
