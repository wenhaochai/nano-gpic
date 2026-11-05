#!/bin/bash
#SBATCH -J gpic-point-exact
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:40:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The sampler control of test E on the real data: w1147's point model, no caption dropout, at C0/16 and C0/8, with the energy head's exact step (deterministic DDIM, no clamp, the last step on x_hat), at 50 and 8 calls, each prompt's own noise. docs/energy_head.md, section 9. Expected: about 20 min. Code: the frozen worktree gpic-wt/energy_real.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/energy_real
[ -f "$WT/baselines/PixelGen/configs_h100/exact_sample.yaml" ] || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/point_nodrop_d640 EVAL_CONFIG=exact_sample
fail=0
for STEP in 33408 66816; do
  bash $WT/jobs/eval_energy.sh 640 9 2 $STEP 50 8 > $LADDER_RUN_ROOT/eval_exact_$STEP.log 2>&1 || fail=1
  echo "eval $STEP fail=$fail"
done
exit $fail
