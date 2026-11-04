#!/bin/bash
#SBATCH -J gpic-noise-eval
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:50:00
#SBATCH --mem=400G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# What does the official noise cost? The official x_T is one tensor for ALL 50K prompts (gpic_jsonl_dataset.py: seed = idx % 1 = 0), so the samples of one evaluation share their noise.
# On the baseline's checkpoints (w1053, d640, official recipe): 6 evaluations, no training, in two 4-GPU jobs. The rest of the protocol is eval_ladder.sh's: EMA net, 50 Euler calls, CFG 1.0, 50K test captions, FD-DINOv2.
#   C0/8:  official (must give w1053's FD 1503.63 within 0.1%, which checks the new code path; w1053 ran on 8 GPUs, so the batches differ, and the text encoder's rounding with them), fixed1 and fixed2 (one other tensor for all prompts: how much the tensor's identity matters),
#          prompt_a and prompt_b (each prompt its own noise: the cost of sharing, and the spread of FD under independent noise)
#   C0/16: prompt_a
# Reading, set before the run: the spread of official, fixed1, fixed2 shows the effect of the tensor's identity. The gap between their mean and the mean of prompt_a, prompt_b is the cost of sharing one noise.
# It keeps the DINOv2 features, and the first 5000 images of the fixed arms. tools/dist_step/noise_share_real.py then gives the share of the variance that comes from the noise (on a vis node, after both jobs).
# usage: eval_noise.sh <arm:step>...    e.g. eval_noise.sh official:066816 fixed1:066816 fixed2:066816
# Expected: about 30 min (9 min per arm on 4 GPUs). Code: the frozen worktree gpic-wt/code_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${LADDER_WT:-$WC/project/gpic-wt/code_run}
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
CKPT_DIR=$WC/runs/gpic-speed/ladder_seed_same/exp_ladder_d640
OUT=$WC/runs/gpic-speed/noise_eval_$(git -C $WT rev-parse --short=7 HEAD)
mkdir -p $OUT
git -C $WT rev-parse HEAD > $OUT/commit.txt
python $WT/tools/gpu_health.py > $OUT/gpu_health.txt 2>&1 || { tail -3 $OUT/gpu_health.txt; echo "unhealthy GPU on $(hostname); exit 98"; exit 98; }
cd $WT/baselines/PixelGen
export PYTHONPATH=$PWD WANDB_MODE=disabled
fail=0
evaluate() {   # <arm> <step> <extra predict arguments...>
  local ARM=$1 STEP=$2; shift 2
  local D=$OUT/${ARM}_$STEP; mkdir -p $D
  local t0=$(date +%s)
  python main.py predict --config configs_h100/lite_ladder_sample.yaml --trainer.devices 4 --trainer.default_root_dir "$D/pred" --ckpt_path $CKPT_DIR/ms-step=$STEP.ckpt \
    --model.diffusion_sampler.init_args.num_steps 50 --model.denoiser.init_args.hidden_size 640 --model.denoiser.init_args.num_groups 10 \
    --model.denoiser.init_args.num_blocks 9 --model.denoiser.init_args.num_text_blocks 2 "$@" > $D/predict.log 2>&1 || { echo "predict failed: $ARM $STEP"; fail=1; return; }
  local NPZ=$(find "$D/pred" -name output.npz | head -1)
  python -m torch.distributed.run --standalone --nproc_per_node 4 $WT/tools/fast_eval.py "$NPZ" $WC/data/gpic/reference_stats/test_stats.npz $D/metrics.json --workers 11 --save-features $D/features.npy > $D/eval.log 2>&1 || { echo "eval failed: $ARM $STEP"; fail=1; return; }
  python - "$NPZ" $D/first.npy $ARM <<'PY'
import sys
import numpy as np
np.save(sys.argv[2], np.load(sys.argv[1])["arr_0"][:5000 if sys.argv[3].startswith("fixed") else 64])
PY
  rm -f "$NPZ"   # 50K x 196 KB
  echo "$ARM $STEP $(python3 -Ic "import json,sys; d=json.load(open(sys.argv[1])); print({k: round(d[k], 4) for k in ('fd','fd_mu','fd_sigma','precision','recall','density','coverage')})" $D/metrics.json) seconds $(($(date +%s) - t0))"
}
for spec in "$@"; do
  IFS=: read -r arm step <<<"$spec"
  case $arm in
    official) evaluate $arm $step ;;
    fixed1) evaluate $arm $step --data.pred_dataset.init_args.noise_seed 1 ;;
    fixed2) evaluate $arm $step --data.pred_dataset.init_args.noise_seed 2 ;;
    prompt_a) evaluate $arm $step --data.pred_dataset.init_args.noise_per_prompt true ;;
    prompt_b) evaluate $arm $step --data.pred_dataset.init_args.noise_per_prompt true --data.pred_dataset.init_args.noise_seed 100000 ;;
    *) echo "unknown arm $arm"; fail=1 ;;
  esac
done
echo "noise eval done fail=$fail $OUT"
exit $fail
