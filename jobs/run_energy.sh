#!/bin/bash
#SBATCH -J gpic-energy
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH --time=02:00:00
#SBATCH --mem=800G
#SBATCH -o /scratch/gpfs/GROUP/USER/project/gpic/logs/%x_%j.out
#SBATCH --comment=fetchable
#FETCHABLE
# Test E on the real data: the energy head, lambda 0.5, at d640 to C0/8, then FD at C0/16 and C0/8 at 50 calls with each prompt's own noise. Same compute as the point model: 1.94% more FLOPs per step, so 66816 / 1.0194 = 65544, a multiple of 32 is 65536 steps, and C0/16 is 32768. No caption dropout. Its baseline is jobs/run_point_nodrop.sh (w1147: 1539.52, 1089.92). Rule, set before the run, in docs/energy_head.md, section 9. Code: the frozen worktree gpic-wt/energy_real. Same shard order and the same official crop as w1147.
set -o pipefail
source /scratch/gpfs/GROUP/USER/project/gpic/jobs/env.sh
WT=$WC/project/gpic-wt/energy_real
[ -d "$WT/baselines/PixelGen" ] || { echo "no worktree $WT"; exit 2; }
grep -q "lam: 0.5" $WT/baselines/PixelGen/configs_h100/energy_train.yaml || { echo "$WT holds old code"; exit 2; }
[ -z "$(git -C $WT status --porcelain --untracked-files=no)" ] || { echo "$WT has local changes"; exit 2; }
ROOT=$WC/runs/gpic-speed/energy_d640
mkdir -p $ROOT
export LADDER_WT=$WT LADDER_RUN_ROOT=$ROOT
export GPIC_SHARD_SEED=1730000000 GPIC_REGION_CROP=0
bash $WT/jobs/train_ladder.sh d640 640 9 2 65536 --config configs_h100/energy_train.yaml --trainer.logger.init_args.name energy_d640 --trainer.logger.init_args.id energy-d640 >> $ROOT/train.log 2>&1
rc=$?
echo "train rc=$rc"
[ $rc -eq 0 ] || exit $rc
fail=0
for STEP in 32768 65536; do
  bash $WT/jobs/eval_energy.sh 640 9 2 $STEP 50 8 > $ROOT/eval_$STEP.log 2>&1 || fail=1
  echo "eval $STEP fail=$fail"
done
exit $fail
