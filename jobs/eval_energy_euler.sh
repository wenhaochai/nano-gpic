#!/bin/bash
#SBATCH -J gpic-energy-euler
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:25:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The energy head on the point model's official Euler at C0/16 and C0/8, 50 calls, each prompt's own noise, no training. docs/energy_head.md, section 11. Expected: about 12 min. Code: the frozen worktree gpic-wt/energy_real.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/energy_real
[ -f "$WT/baselines/PixelGen/configs_h100/energy_euler_sample.yaml" ] || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
export LADDER_WT=$WT LADDER_RUN_ROOT=$WC/runs/gpic-speed/energy_d640 EVAL_CONFIG=energy_euler_sample EVAL_SUF=_euler
fail=0
for STEP in 32768 65536; do
  bash $WT/jobs/eval_energy.sh 640 9 2 $STEP 50 > $LADDER_RUN_ROOT/eval_euler_$STEP.log 2>&1 || fail=1
  echo "eval $STEP fail=$fail"
done
exit $fail
