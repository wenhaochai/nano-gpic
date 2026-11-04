#!/bin/bash
#SBATCH -J gpic-stoch-eval
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:50:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Random sampling at each step, on the same weights (w1053, d640, official recipe, EMA), no training. The network's mean plus a known variance is a Gaussian distribution per step: the simplest form of the owner's idea (docs/distribution_step.md 5.7).
# The protocol is eval_noise.sh's: 50 calls on the official grid, CFG 1.0, 50K test captions, FD-DINOv2. Each prompt has its own x_T except in the official arm.
#   C0/8:  eta1 (the DDPM posterior at each step), eta05 (half of its fresh noise), land (Euler, and the last step lands on x_hat, which removes the clamp's leftover noise), eta1_official (eta 1, the official x_T)
#   C0/16: eta1
# Reading, set before the run. Baseline: Euler with each prompt's own noise, FD 1083.94 at C0/8, 1529.61 at C0/16 (w1098, w1099). Two draws differ by 0.007%, so the evaluation noise is small.
# A random sampler counts as a gain only if FD is lower by more than 5% at both points, the bar of plan A. land is a separate question: it needs no randomness.
# Expected: about 30 min (about 6 min per arm on 4 GPUs). Code: the frozen worktree gpic-wt/stoch_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/stoch_run}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
CKPT_DIR=$WC/runs/gpic-speed/ladder_seed_same/exp_ladder_d640
OUT=$WC/runs/gpic-speed/stoch_eval_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled
PROMPT="--data.pred_dataset.init_args.noise_per_prompt true"
fail=0
evaluate() {   # <arm> <step> <extra predict arguments...>
  local ARM=$1 STEP=$2; shift 2
  local D=$OUT/${ARM}_$STEP; mkdir -p $D
  local t0=$(date +%s)
  python main.py predict --config configs_h100/lite_ladder_sample.yaml --trainer.devices 4 --trainer.default_root_dir "$D/pred" --ckpt_path $CKPT_DIR/ms-step=$STEP.ckpt \
    --model.diffusion_sampler.init_args.num_steps 50 --model.denoiser.init_args.hidden_size 640 --model.denoiser.init_args.num_groups 10 \
    --model.denoiser.init_args.num_blocks 9 --model.denoiser.init_args.num_text_blocks 2 "$@" > $D/predict.log 2>&1 || { echo "predict failed: $ARM $STEP"; fail=1; return; }
  local NPZ=$(find "$D/pred" -name output.npz | head -1)
  python -m torch.distributed.run --standalone --nproc_per_node 4 $WT/tools/fast_eval.py "$NPZ" $WC/data/gpic/reference_stats/test_stats.npz $D/metrics.json --workers 11 > $D/eval.log 2>&1 || { echo "eval failed: $ARM $STEP"; fail=1; return; }
  python - "$NPZ" $D/first.npy <<'PY'
import sys
import numpy as np
np.save(sys.argv[2], np.load(sys.argv[1])["arr_0"][:64])
PY
  rm -f "$NPZ"   # 50K x 196 KB
  echo "$ARM $STEP $(python3 -Ic "import json,sys; d=json.load(open(sys.argv[1])); print({k: round(d[k], 4) for k in ('fd','fd_mu','fd_sigma','precision','recall','density','coverage')})" $D/metrics.json) seconds $(($(date +%s) - t0))"
}
evaluate eta1 066816 $PROMPT --model.diffusion_sampler.init_args.eta 1.0
evaluate eta05 066816 $PROMPT --model.diffusion_sampler.init_args.eta 0.5
evaluate land 066816 $PROMPT --model.diffusion_sampler.init_args.land true
evaluate eta1_official 066816 --model.diffusion_sampler.init_args.eta 1.0
evaluate eta1 033408 $PROMPT --model.diffusion_sampler.init_args.eta 1.0
echo "stoch eval done fail=$fail $OUT"
exit $fail
