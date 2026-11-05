#!/bin/bash
#SBATCH -J gpic-sample-grid
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --time=00:25:00
#SBATCH --mem=200G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Samples for the page (question 10): the first 8 test captions, each prompt's own noise, no CFG, through the production predict path, from 4 checkpoints at an eighth of the compute: the point model (w1147), the energy head on the official Euler (w1159), autoregression with eta 20 (w1151) and with the first 64 drawn again (w1150). One GPU each. Expected: about 6 min. Code: the frozen worktrees ar_real and energy_real.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
R=$WC/runs/gpic-speed
OUT=$R/sample_grid_$(date +%m%d_%H%M)
mkdir -p $OUT
head -n 8 $WC/data/gpic/gpic_test_50k.jsonl > $OUT/first8.jsonl
IFS=, read -r -a DEV <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
(( ${#DEV[@]} >= 4 )) || { echo "${#DEV[@]} GPUs"; exit 2; }
one() {   # gpu name worktree ckpt [extra predict arguments]
  local g=$1 name=$2 wt=$3 ckpt=$4; shift 4
  ( cd $WC/project/gpic-wt/$wt/baselines/PixelGen && export PYTHONPATH=$PWD WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${DEV[$g]} && \
    timeout -k 60 1200 python main.py predict --config configs_h100/lite_ladder_sample.yaml "$@" --trainer.devices 1 \
      --trainer.default_root_dir $OUT/$name --ckpt_path $ckpt \
      --model.denoiser.init_args.hidden_size 640 --model.denoiser.init_args.num_groups 10 \
      --model.denoiser.init_args.num_blocks 9 --model.denoiser.init_args.num_text_blocks 2 \
      --data.pred_dataset.init_args.jsonl_path $OUT/first8.jsonl --data.pred_dataset.init_args.noise_per_prompt true ) > $OUT/$name.log 2>&1
}
one 0 point ar_real $R/point_nodrop_d640/exp_ladder_d640/ms-step=066816.ckpt &
one 1 energy energy_real $R/energy_d640/exp_ladder_d640/ms-step=065536.ckpt --config configs_h100/energy_euler_sample.yaml &
one 2 ar_eta20 ar_real $R/ar_few_d640/exp_ladder_d640/ms-step=052320.ckpt --config configs_h100/ar_sample.yaml --model.denoiser.init_args.eta 20 &
one 3 ar_first64 ar_real $R/ar_mar_d640/exp_ladder_d640/ms-step=052320.ckpt --config configs_h100/ar_sample.yaml --model.denoiser.init_args.redraw_first 64 &
fail=0
for p in $(jobs -p); do wait $p || fail=1; done
for n in point energy ar_eta20 ar_first64; do echo "$n: $(find $OUT/$n -name output.npz | head -1)"; tail -n 2 $OUT/$n.log | cut -c1-200; done
echo "grid done fail=$fail $OUT"
exit $fail
