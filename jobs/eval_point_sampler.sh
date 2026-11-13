#!/bin/bash
#SBATCH -J gpic-point-gmsde
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=00:45:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# The sampler control of GMFlow (docs/gmflow.md): w1147's point model (no caption dropout; runs/gpic-speed/point_nodrop_d640, logs/point_nodrop.yaml) at C0/16 and C0/8, with
# GM-SDE read as K = 1, s = 0, which is DDPM with the small variance, at 50 and 8 calls, and the official Euler at 8 calls (its 50-call numbers are 1539.52, 1089.92).
# Each prompt's own noise, EMA net, no CFG. Expected: about 30 min. Code: the frozen worktree gpic-wt/gmflow_run. Output: runs/gpic-speed/point_sampler_gmflow, with
# links to w1147's checkpoints, so nothing is written in w1147's run directory (review F6).
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/gmflow_run
export GMFLOW_WT=$WT
[ -f "$WT/baselines/PixelGen/src/diffusion/flow_matching/sampling_gmflow.py" ] || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
SRC=$WC/runs/gpic-speed/point_nodrop_d640/exp_ladder_d640
ROOT=$WC/runs/gpic-speed/point_sampler_gmflow
mkdir -p $ROOT/exp_ladder_d640
for STEP in 033408 066816; do [ -e $ROOT/exp_ladder_d640/ms-step=$STEP.ckpt ] || ln -s $SRC/ms-step=$STEP.ckpt $ROOT/exp_ladder_d640/; done
git -C $WT rev-parse HEAD > $ROOT/commit.txt
fail=0
for STEP in 33408 66816; do
  for SC in gm_sde:50 gm_sde:8 euler:8; do
    S=${SC%:*} C=${SC#*:}
    if [ -f $ROOT/exp_ladder_d640/eval_$(printf %06d $STEP)_${S}_calls${C}/metrics.json ]; then echo "eval $STEP $S $C done before"; continue; fi
    LADDER_RUN_ROOT=$ROOT bash $WT/jobs/eval_gmflow.sh 0 $STEP $S $C > $ROOT/eval_${STEP}_${S}_${C}.log 2>&1 || fail=1
    echo "eval $STEP $S $C fail=$fail"
  done
done
exit $fail
