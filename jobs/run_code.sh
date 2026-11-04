#!/bin/bash
#SBATCH -J gpic-code
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=01:50:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Plan A at d640 to C0/8 at one rate, then FD at C0/16 and C0/8 (docs/distribution_step.md 5.2). Baseline: w1053 (same shard order): FD 2099 at C0/16, 1504 at C0/8.
# Same FLOPs: 34.7 GFLOP forward + 0.34 for the encoder and the flow (tools/dist_step/test_code.py prints it), so 66816 x 104.15 / 105.17 = 66168, and 66176 steps (a multiple of 32, with a quarter a whole number).
# Valid only if FD is below the baseline at both points by more than 5%, set before the run: the seed test shows 2-4% between two runs of one recipe.
# usage: run_code.sh <R, nats>     Code: the frozen worktree gpic-wt/code_run.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
R=$1
[ -n "$R" ] || { echo "usage: run_code.sh <R>"; exit 2; }
WT=$WC/project/gpic-wt/code_run
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
ROOT=$WC/runs/gpic-speed/code_r$R
mkdir -p $ROOT
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT
export GPIC_SHARD_SEED=1730000000   # the shard order of the baseline, w1053
bash $WT/jobs/train_ladder.sh d640 640 9 2 66176 --config configs_h100/code_train.yaml --model.denoiser.init_args.code_rate $R \
  --trainer.logger.init_args.name code_r$R --trainer.logger.init_args.id code-r$R >> $ROOT/train.log 2>&1
rc=$?
echo "train rc=$rc"
[ $rc -eq 0 ] || exit $rc
fail=0
for STEP in 33088 66176; do
  bash $WT/jobs/eval_code.sh d640 640 9 2 $STEP 50 > $ROOT/eval_$STEP.log 2>&1 || fail=1
  echo "eval $STEP fail=$fail"
done
exit $fail
