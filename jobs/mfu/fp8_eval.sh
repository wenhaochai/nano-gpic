#!/bin/bash
#SBATCH -J gpic-mfu-fp8eval
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Train FP8, evaluate FP8 (owner 2026-11-08: 「不应该train什么eval什么吗」, 「做」). The d16 FP8 check (runs/gpic-speed/muon_fp8recipe)
# scored both arms in bf16: FP8 lost FD +9.9%, FID +13.0%, held-out +0.10% at C0. Here both arms' C0 (and the 10200) checkpoints are scored
# through the training's FP8 GEMMs as well:
# 1. held-out loss, 8 tasks side by side (tools/mfu/heldout_fp8.py): C0 of each arm as raw (the tool's own number), bf16w and fp8;
#    10200 of each arm as fp8.
# 2. FD-DINOv2 and FID of each arm's C0 with GPIC_EVAL_FP8=native2 (branch eval-fp8 592f089, worktree gpic-wt/eval_fp8_run; the same
#    raw views as the bf16 evaluations, linked into runs/gpic-speed/fp8_eval/<arm>), one 8-GPU evaluation after the other.
# usage: fp8_eval.sh <mfu worktree>
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=${1:?mfu worktree}
export NCCL_NVLS_ENABLE=0
R=$WC/runs/gpic-speed/muon_fp8recipe
O=$WC/runs/gpic-speed/fp8_eval
mkdir -p $O
J=$O/heldout_fp8.jsonl
T=(
  "$R/fp8-d20-n3/exp_ladder_d1536/ms-step=013600.ckpt raw"
  "$R/fp8-d20-n3/exp_ladder_d1536/ms-step=013600.ckpt fp8"
  "$R/fp8-d20-n3/exp_ladder_d1536/ms-step=013600.ckpt bf16w"
  "$R/fp8-d20-bf16/exp_ladder_d1536/ms-step=013600.ckpt raw"
  "$R/fp8-d20-bf16/exp_ladder_d1536/ms-step=013600.ckpt fp8"
  "$R/fp8-d20-bf16/exp_ladder_d1536/ms-step=013600.ckpt bf16w"
  "$R/fp8-d20-n3/exp_ladder_d1536/ms-step=010200.ckpt fp8"
  "$R/fp8-d20-bf16/exp_ladder_d1536/ms-step=010200.ckpt fp8"
)
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g python $WT/tools/mfu/heldout_fp8.py ${T[$g]} $J > $O/heldout_g$g.log 2>&1 &
done
wait
grep -h "HELDOUT_FP8\|Error\|Traceback" $O/heldout_g*.log
export GPIC_EVAL_FP8=native2
for arm in fp8-d20-n3 fp8-d20-bf16; do
  bash $WC/project/gpic-wt/eval_fp8_run/jobs/evals_gmflow.sh 0 d1536 1536 16 4 $O/$arm \
    --link $WC/runs/gpic-speed/raw_view/muon_fp8recipe/$arm/exp_ladder_d1536 --wt eval_fp8_run 13600:euler:50
  echo "$arm rc=$?"
  cat $O/$arm/exp_ladder_d1536/eval_013600_euler_calls50_official/metrics.json 2>/dev/null | head -8
  cat $O/$arm/exp_ladder_d1536/eval_013600_euler_calls50_official/metrics_inception.json 2>/dev/null | head -3
done
echo FP8_EVAL_DONE
