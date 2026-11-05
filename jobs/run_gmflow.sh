#!/bin/bash
#SBATCH -J gpic-gmflow
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=03:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# GMFlow on the real data (docs/gmflow.md): the Gaussian-mixture head with the transition loss (lambda 0.5) at d640 to C0/8, then FD at C0/16 and C0/8 with each prompt's own
# noise, for GM-SDE, GM-SDE with the mixture mean on its last step, GM-ODE and the official Euler on the mixture mean, at 50 and at 8 calls (owner 2026-11-05: the two
# call counts are judged apart). 16 evaluations; the time is 03:00 (review F3).
# Time: training about 36 min x (1 + r) plus the head's fp32 loss (w1147 trained in 36 min), 12 evaluations of about 2 to 5 min each; 02:45 leaves room. A rerun skips finished parts. Same compute as the point model w1147 (66816 steps): STEPS is the largest multiple of 128
# with STEPS (1 + r) <= 66816, r = the added forward FLOPs / 34.72 GFLOP (FlopCounterMode, tools/gmflow/test_gmflow.py T9). No caption dropout, same shard order and crop as w1147.
# usage: run_gmflow.sh <K: 8 | 1>      K = 1 is the control that isolates the mixture from the change of loss weighting (digest section 12, note 4).
# Other samplers and call counts afterwards: jobs/eval_gmflow.sh. Code: the frozen worktree gpic-wt/gmflow_run (review F2); the run records its commit and every
# evaluation checks that the worktree still holds it.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
K=$1
case $K in   # K: r, STEPS
  8) STEPS=62336 ;;   # r = 0.07010, 62336 x 1.07010 = 66706
  1) STEPS=66560 ;;   # r = 0.00245, 66560 x 1.00245 = 66723
  *) echo "usage: run_gmflow.sh <8 | 1>"; exit 2 ;;
esac
MS=$((STEPS / 4))   # ms checkpoints at STEPS/4, /2, 3/4 and the end (train_ladder.sh); C0/16 = 2 MS
WT=$WC/project/gpic-wt/gmflow_run
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q "lam: 0.5" $WT/baselines/PixelGen/configs_h100/gmflow_train.yaml || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
ROOT=$WC/runs/gpic-speed/gmflow_k${K}_d640
mkdir -p $ROOT
SHA=$(git -C $WT rev-parse HEAD)
if [ -f $ROOT/commit.txt ] && [ "$(cat $ROOT/commit.txt)" != "$SHA" ]; then echo "$ROOT was started at $(cat $ROOT/commit.txt), the worktree holds $SHA"; exit 2; fi
echo $SHA > $ROOT/commit.txt
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT GMFLOW_WT=$WT
export GPIC_SHARD_SEED=1730000000 GPIC_REGION_CROP=0
bash $WT/jobs/train_ladder.sh d640 640 9 2 $STEPS --config configs_h100/gmflow_train.yaml --model.denoiser.init_args.gm_k $K \
  --trainer.logger.init_args.name gmflow_k${K}_d640 --trainer.logger.init_args.id gmflow-k${K}-d640 >> $ROOT/train.log 2>&1
rc=$?
echo "train rc=$rc launches=$(wc -l < $ROOT/exp_ladder_d640/launches.txt 2>/dev/null)"   # more than 1: the run was cut and resumed (review F7)
[ $rc -eq 0 ] || exit $rc
fail=0
for STEP in $((2 * MS)) $STEPS; do
  for C in 50 8; do
    for S in gm_sde gm_sde_final_mean gm_ode euler_mean; do
      D=$ROOT/exp_ladder_d640/eval_$(printf %06d $STEP)_${S}_calls$C
      if [ -f $D/metrics.json ]; then echo "eval $STEP $S $C done before"; continue; fi
      [ "$(git -C $WT rev-parse HEAD)" = "$SHA" ] || { echo "$WT moved from $SHA"; exit 2; }
      bash $WT/jobs/eval_gmflow.sh $K $STEP $S $C > $ROOT/eval_${STEP}_${S}_$C.log 2>&1 || fail=1
      echo "eval $STEP $S $C fail=$fail"
    done
  done
done
exit $fail
